"""
TaxRagEngineV2 - Async RAG engine for ATO tax document Q&A.

Uses LangChain with Google Gemini or OpenAI for embeddings and LLM.
Supports both ChromaDB (local) and Pinecone (cloud) vector stores.
Features async ingestion with concurrent batch processing.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Callable

from dotenv import load_dotenv
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


@dataclass
class AskResult:
    """Result from asking a question."""

    answer: str
    source_documents: list
    scores: list  # Similarity scores (distance - lower is better)


@dataclass
class IngestResult:
    """Result from ingesting documents."""

    total_chunks: int
    batches_processed: int
    elapsed_time: float


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff."""

    max_attempts: int = 7
    min_wait: float = 5.0    # Start with 5s for rate limit recovery
    max_wait: float = 120.0  # Max 2 min wait
    exp_base: float = 2.0    # 5s → 10s → 20s → 40s → 80s → 120s


class TaxRagEngineV2:
    """Async RAG engine for Australian Tax Office document Q&A."""

    def __init__(
        self,
        model_provider: Literal["google", "openai"] = "google",
        persist_directory: str = "./chroma_db",
    ):
        """
        Initialize the RAG engine.

        Args:
            model_provider: LLM provider - "google" or "openai"
            persist_directory: Directory to persist the local vector store
        """
        self.model_provider = model_provider
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.vector_store_type = os.getenv("VECTOR_STORE_TYPE", "chroma")

        self._validate_environment()
        self._init_llm()
        self._init_embeddings()
        self._init_text_splitter()

    def _validate_environment(self) -> None:
        """Validate required environment variables are set."""
        if self.model_provider == "google":
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError(
                    "GOOGLE_API_KEY environment variable is required for Google provider. "
                    "Get your API key from https://aistudio.google.com/apikey"
                )
        elif self.model_provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required for OpenAI provider."
                )
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

        if self.vector_store_type == "pinecone":
            if not os.getenv("PINECONE_API_KEY"):
                raise ValueError(
                    "PINECONE_API_KEY environment variable is required for Pinecone vector store. "
                    "Get your API key from https://www.pinecone.io"
                )

    def _init_llm(self) -> None:
        """Initialize the LLM based on provider."""
        if self.model_provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3,
            )
        else:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.3,
            )

    def _init_embeddings(self) -> None:
        """Initialize the embedding model based on provider with SDK retries disabled."""
        if self.model_provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from google.genai.types import HttpRetryOptions

            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                output_dimensionality=768,  # Match existing Pinecone index
                request_options={"retry": HttpRetryOptions(attempts=1)},  # Disable SDK retry
            )
        else:
            from langchain_openai import OpenAIEmbeddings

            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                max_retries=0,  # Disable SDK retry
            )

    def _init_text_splitter(self) -> None:
        """Initialize the text splitter."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    # ============== RETRY METHODS ==============

    def _get_retry_exceptions(self) -> tuple:
        """Get retryable exceptions for current provider (429 + 5xx + transient)."""
        # RuntimeError for async session timeouts (e.g., "Session is closed")
        base_exceptions = [RuntimeError]

        if self.model_provider == "google":
            from langchain_google_genai._common import GoogleGenerativeAIError

            return tuple(base_exceptions + [GoogleGenerativeAIError])
        else:
            from openai import RateLimitError, APIError, APIConnectionError

            return tuple(base_exceptions + [RateLimitError, APIError, APIConnectionError])

    def _create_retryer(self, config: RetryConfig | None = None) -> AsyncRetrying:
        """Factory to create async retryer for all providers."""
        config = config or RetryConfig()

        return AsyncRetrying(
            stop=stop_after_attempt(config.max_attempts),
            wait=wait_exponential(
                multiplier=1,
                min=config.min_wait,
                max=config.max_wait,
                exp_base=config.exp_base,
            ),
            retry=retry_if_exception_type(self._get_retry_exceptions()),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    # ============== ASYNC METHODS ==============

    async def _aload_pdf(self, pdf_path: str) -> list:
        """Async load PDF documents."""
        loader = PyPDFLoader(str(pdf_path))
        documents = await loader.aload()
        return documents

    async def _asplit_documents(self, documents: list) -> list:
        """
        Run sync text splitter in thread pool executor.
        RecursiveCharacterTextSplitter has no async version.
        """
        loop = asyncio.get_event_loop()
        splits = await loop.run_in_executor(
            None,
            self.text_splitter.split_documents,
            documents,
        )
        return splits

    async def _aingest_batch_with_retry(
        self,
        batch: list,
        batch_num: int,
        total_batches: int,
        retry_config: RetryConfig | None = None,
    ) -> list[str]:
        """
        Ingest a batch with exponential backoff retry.

        Args:
            batch: Documents to ingest
            batch_num: Current batch number (for logging)
            total_batches: Total number of batches (for logging)
            retry_config: Retry configuration

        Returns:
            List of document IDs
        """
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        retryer = self._create_retryer(retry_config)

        async for attempt in retryer:
            with attempt:
                ids = await self.vectorstore.aadd_documents(batch)
                return ids

    async def _ainit_vectorstore_with_batch(self, first_batch: list) -> None:
        """Initialize vector store with first batch of documents."""
        if self.vector_store_type == "pinecone":
            from langchain_pinecone import PineconeVectorStore

            index_name = os.getenv("PINECONE_INDEX_NAME", "ato-tax-assistant")

            self.vectorstore = await PineconeVectorStore.afrom_documents(
                documents=first_batch,
                embedding=self.embeddings,
                index_name=index_name,
            )
        else:
            from langchain_community.vectorstores import Chroma

            # Chroma.afrom_documents may not be fully async, run in executor
            loop = asyncio.get_event_loop()
            self.vectorstore = await loop.run_in_executor(
                None,
                lambda: Chroma.from_documents(
                    documents=first_batch,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                ),
            )

    async def aingest_pdf(
        self,
        pdf_path: str,
        batch_size: int = 50,
        max_concurrency: int = 3,
        retry_config: RetryConfig | None = None,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> IngestResult:
        """
        Async ingest PDF with concurrent batch processing and retry.

        Args:
            pdf_path: Path to the PDF file
            batch_size: Number of documents per batch
            max_concurrency: Maximum concurrent batch operations
            retry_config: Retry configuration for handling 429/5xx errors
            progress_callback: Optional callback(batch_num, total_batches, chunk_count)

        Returns:
            IngestResult with statistics
        """
        start_time = time.time()

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        print(f"Loading PDF: {pdf_path}")
        print(f"Vector store type: {self.vector_store_type}")
        print(f"Max concurrency: {max_concurrency}")

        # Step 1: Async load PDF
        print("Step 1/4: Loading PDF...")
        documents = await self._aload_pdf(str(pdf_path))
        print(f"  Loaded {len(documents)} pages")

        # Step 2: Split documents (in executor, CPU-bound)
        print("Step 2/4: Splitting documents...")
        splits = await self._asplit_documents(documents)
        total_chunks = len(splits)
        print(f"  Created {total_chunks} chunks")

        # Step 3: Initialize vectorstore with first batch
        print("Step 3/4: Initializing vector store...")
        first_batch = splits[:batch_size]
        remaining = splits[batch_size:]

        await self._ainit_vectorstore_with_batch(first_batch)
        print(f"  Initialized with {len(first_batch)} chunks")

        # Step 4: Process remaining batches concurrently with retry
        if remaining:
            print(f"Step 4/4: Processing {len(remaining)} remaining chunks...")
            semaphore = asyncio.Semaphore(max_concurrency)

            batches = [
                remaining[i : i + batch_size]
                for i in range(0, len(remaining), batch_size)
            ]
            total_batches = len(batches)

            async def process_batch(batch: list, batch_num: int) -> list[str]:
                async with semaphore:
                    return await self._aingest_batch_with_retry(
                        batch, batch_num, total_batches + 1, retry_config
                    )

            tasks = [
                process_batch(batch, i + 2)  # +2 because first batch already processed
                for i, batch in enumerate(batches)
            ]
            await asyncio.gather(*tasks)

            batches_processed = total_batches + 1
        else:
            batches_processed = 1

        elapsed_time = time.time() - start_time
        print(f"\nCompleted in {elapsed_time:.2f}s")

        return IngestResult(
            total_chunks=total_chunks,
            batches_processed=batches_processed,
            elapsed_time=elapsed_time,
        )

    # ============== SYNC METHODS (for backwards compatibility) ==============

    def _init_pinecone(self) -> None:
        """Initialize Pinecone vector store."""
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone

        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "ato-tax-assistant")

        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        self.vectorstore = PineconeVectorStore(
            index=index,
            embedding=self.embeddings,
        )

    def _init_chroma(self) -> None:
        """Initialize ChromaDB vector store from disk."""
        from langchain_community.vectorstores import Chroma

        if not Path(self.persist_directory).exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.persist_directory}. "
                "Please run ingest_pdf() first."
            )

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

    def load_vectorstore(self) -> None:
        """Load vector store based on VECTOR_STORE_TYPE environment variable."""
        if self.vector_store_type == "pinecone":
            self._init_pinecone()
        else:
            self._init_chroma()

    def ingest_pdf(self, pdf_path: str, batch_size: int = 50, delay: float = 1.0) -> int:
        """
        Sync ingest PDF (backwards compatible).
        For better performance, use aingest_pdf() instead.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        splits = self.text_splitter.split_documents(documents)

        total_chunks = len(splits)
        print(f"Processing {total_chunks} chunks in batches of {batch_size}...")
        print(f"Vector store type: {self.vector_store_type}")

        if self.vector_store_type == "pinecone":
            self._ingest_to_pinecone_sync(splits, batch_size, delay)
        else:
            self._ingest_to_chroma_sync(splits, batch_size, delay)

        return total_chunks

    def _ingest_to_chroma_sync(self, splits: list, batch_size: int, delay: float) -> None:
        """Sync ingest to ChromaDB."""
        from langchain_community.vectorstores import Chroma

        total_chunks = len(splits)

        for i in range(0, total_chunks, batch_size):
            batch = splits[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

            if i == 0:
                self.vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                )
            else:
                self.vectorstore.add_documents(batch)

            if i + batch_size < total_chunks:
                time.sleep(delay)

    def _ingest_to_pinecone_sync(self, splits: list, batch_size: int, delay: float) -> None:
        """Sync ingest to Pinecone."""
        from langchain_pinecone import PineconeVectorStore

        index_name = os.getenv("PINECONE_INDEX_NAME", "ato-tax-assistant")
        total_chunks = len(splits)

        for i in range(0, total_chunks, batch_size):
            batch = splits[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

            if i == 0:
                self.vectorstore = PineconeVectorStore.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    index_name=index_name,
                )
            else:
                self.vectorstore.add_documents(batch)

            if i + batch_size < total_chunks:
                time.sleep(delay)

    def ask(self, question: str, k: int = 4) -> AskResult:
        """
        Ask a question about the ingested documents.

        Args:
            question: The question to ask
            k: Number of relevant documents to retrieve

        Returns:
            AskResult containing answer, source documents, and similarity scores
        """
        if self.vectorstore is None:
            self.load_vectorstore()

        results_with_scores = self.vectorstore.similarity_search_with_score(
            question, k=k
        )
        docs = [doc for doc, _ in results_with_scores]
        scores = [score for _, score in results_with_scores]

        context = "\n\n".join(
            f"[Page {doc.metadata.get('page', 'N/A')}]: {doc.page_content}"
            for doc in docs
        )

        system_prompt = """You are an expert assistant for Australian tax matters.
Use the following context from ATO (Australian Taxation Office) documents to answer the question.
If you don't know the answer based on the context, say so clearly.
Always cite relevant sections or page numbers when possible.

Context:
{context}
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        chain = prompt | self.llm
        response = chain.invoke({"context": context, "input": question})

        return AskResult(
            answer=response.content,
            source_documents=docs,
            scores=scores,
        )

    async def aask(self, question: str, k: int = 4) -> AskResult:
        """
        Async ask a question about the ingested documents.

        Args:
            question: The question to ask
            k: Number of relevant documents to retrieve

        Returns:
            AskResult containing answer, source documents, and similarity scores
        """
        if self.vectorstore is None:
            self.load_vectorstore()

        results_with_scores = await self.vectorstore.asimilarity_search_with_score(
            question, k=k
        )
        docs = [doc for doc, _ in results_with_scores]
        scores = [score for _, score in results_with_scores]

        context = "\n\n".join(
            f"[Page {doc.metadata.get('page', 'N/A')}]: {doc.page_content}"
            for doc in docs
        )

        system_prompt = """You are an expert assistant for Australian tax matters.
Use the following context from ATO (Australian Taxation Office) documents to answer the question.
If you don't know the answer based on the context, say so clearly.
Always cite relevant sections or page numbers when possible.

Context:
{context}
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        chain = prompt | self.llm
        response = await chain.ainvoke({"context": context, "input": question})

        return AskResult(
            answer=response.content,
            source_documents=docs,
            scores=scores,
        )

    def get_vector_store_type(self) -> str:
        """Return the current vector store type."""
        return self.vector_store_type


async def main():
    """Test async ingestion."""
    engine = TaxRagEngineV2(model_provider="google")

    print(f"Vector store type: {engine.get_vector_store_type()}")

    pdf_path = "./data/tax_guide.pdf"
    if Path(pdf_path).exists():
        print(f"\n{'='*50}")
        print("ASYNC INGESTION TEST")
        print(f"{'='*50}\n")

        result = await engine.aingest_pdf(
            pdf_path,
            batch_size=50,
            max_concurrency=3,
        )

        print(f"\nIngestion Result:")
        print(f"  Total chunks: {result.total_chunks}")
        print(f"  Batches processed: {result.batches_processed}")
        print(f"  Elapsed time: {result.elapsed_time:.2f}s")

        print(f"\n{'='*50}")
        print("ASYNC QUERY TEST")
        print(f"{'='*50}\n")

        question = "What is the tax-free threshold for individuals?"
        print(f"Question: {question}")

        answer_result = await engine.aask(question)
        print(f"Answer: {answer_result.answer[:300]}...")
        print(f"Sources: {len(answer_result.source_documents)} documents")
    else:
        print(f"PDF not found at {pdf_path}")


if __name__ == "__main__":
    asyncio.run(main())
