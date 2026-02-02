"""
TaxRagEngine - RAG engine for ATO tax document Q&A.

Uses LangChain with Google Gemini or OpenAI for embeddings and LLM.
"""

import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


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
            persist_directory: Directory to persist the vector store
        """
        self.model_provider = model_provider
        self.persist_directory = persist_directory
        self.vectorstore = None

        self._validate_environment()
        self._init_llm()
        self._init_embeddings()

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
                model="models/text-embedding-004"
            )
        else:
            from langchain_openai import OpenAIEmbeddings

            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def ingest_pdf(self, pdf_path: str, batch_size: int = 50, delay: float = 1.0) -> int:
        """
        Load and ingest a PDF into the vector store.

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

        # Process in batches to avoid rate limits
        total_chunks = len(splits)
        print(f"Processing {total_chunks} chunks in batches of {batch_size}...")

        for i in range(0, total_chunks, batch_size):
            batch = splits[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

            if i == 0:
                # First batch: create the vector store
                self.vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                )
            else:
                # Subsequent batches: add to existing store
                self.vectorstore.add_documents(batch)

            # Delay between batches to avoid rate limits (skip after last batch)
            if i + batch_size < total_chunks:
                time.sleep(delay)

        return total_chunks

    def load_vectorstore(self) -> None:
        """Load an existing vector store from disk."""
        if not Path(self.persist_directory).exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.persist_directory}. "
                "Please run ingest_pdf() first."
            )

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

    def ask(self, question: str, k: int = 4) -> str:
        """
        Ask a question about the ingested documents.

        Args:
            question: The question to ask
            k: Number of relevant documents to retrieve

        Returns:
            Answer string from the LLM
        """
        if self.vectorstore is None:
            self.load_vectorstore()

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
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

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = retrieval_chain.invoke({"input": question})
        return response["answer"]


if __name__ == "__main__":
    engine = TaxRagEngine(model_provider="google")

    pdf_path = "./data/tax_guide.pdf"
    if Path(pdf_path).exists():
        print(f"Ingesting {pdf_path}...")
        num_chunks = engine.ingest_pdf(pdf_path)
        print(f"Created {num_chunks} document chunks")

        question = "What is the tax-free threshold for individuals?"
        print(f"\nQuestion: {question}")
        answer = engine.ask(question)
        print(f"Answer: {answer}")
    else:
        print(f"PDF not found at {pdf_path}")
