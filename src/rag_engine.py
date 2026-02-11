"""
TaxRagEngine - RAG engine for ATO tax document Q&A.

Uses LangChain with Google Gemini or OpenAI for embeddings and LLM.
Supports both ChromaDB (local) and Pinecone (cloud) vector stores.
"""

import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
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


class TaxRagEngine:
    """RAG engine for Australian Tax Office document Q&A."""

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

    def _validate_environment(self) -> None:
        """Validate required environment variables are set."""
        # Validate LLM provider
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

        # Validate Pinecone if using cloud
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
        """Initialize the embedding model based on provider."""
        if self.model_provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                output_dimensionality=768,  # Match existing Pinecone index
            )
        else:
            from langchain_openai import OpenAIEmbeddings

            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

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
        Load and ingest a PDF into the vector store.

        For ChromaDB: Creates local persistent store.
        For Pinecone: Uploads to cloud index.

        Args:
            pdf_path: Path to the PDF file
            batch_size: Number of documents to process per batch (for rate limiting)
            delay: Delay in seconds between batches (for rate limiting)

        Returns:
            Number of document chunks created
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        splits = text_splitter.split_documents(documents)

        total_chunks = len(splits)
        print(f"Processing {total_chunks} chunks in batches of {batch_size}...")
        print(f"Vector store type: {self.vector_store_type}")

        if self.vector_store_type == "pinecone":
            self._ingest_to_pinecone(splits, batch_size, delay)
        else:
            self._ingest_to_chroma(splits, batch_size, delay)

        return total_chunks

    def _ingest_to_chroma(self, splits: list, batch_size: int, delay: float) -> None:
        """Ingest documents to local ChromaDB."""
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

    def _ingest_to_pinecone(self, splits: list, batch_size: int, delay: float) -> None:
        """Ingest documents to Pinecone cloud."""
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone

        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "ato-tax-assistant")

        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

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

        # Get documents with similarity scores directly
        results_with_scores = self.vectorstore.similarity_search_with_score(
            question, k=k
        )
        docs = [doc for doc, _ in results_with_scores]
        scores = [score for _, score in results_with_scores]

        # Format context from documents
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

        # Invoke LLM directly with formatted context
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "input": question})

        return AskResult(
            answer=response.content,
            source_documents=docs,
            scores=scores,
        )

    def get_vector_store_type(self) -> str:
        """Return the current vector store type."""
        return self.vector_store_type


if __name__ == "__main__":
    engine = TaxRagEngine(model_provider="google")

    print(f"Vector store type: {engine.get_vector_store_type()}")

    pdf_path = "./data/tax_guide.pdf"
    if Path(pdf_path).exists():
        print(f"Ingesting {pdf_path}...")
        num_chunks = engine.ingest_pdf(pdf_path)
        print(f"Created {num_chunks} document chunks")

        question = "What is the tax-free threshold for individuals?"
        print(f"\nQuestion: {question}")
        result = engine.ask(question)
        print(f"Answer: {result.answer}")
        print(f"\nSources: {len(result.source_documents)} documents")
    else:
        print(f"PDF not found at {pdf_path}")
