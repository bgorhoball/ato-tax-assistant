#!/usr/bin/env python3
"""
One-time script to ingest PDF documents to Pinecone.

Before running:
1. Create a free Pinecone account at https://www.pinecone.io
2. Create an index named 'ato-tax-assistant' with dimension 768 (for Google embeddings)
3. Set environment variables:
   - PINECONE_API_KEY=your-api-key
   - PINECONE_INDEX_NAME=ato-tax-assistant (optional, defaults to this)
   - GOOGLE_API_KEY=your-google-api-key
   - VECTOR_STORE_TYPE=pinecone

Usage:
    python scripts/ingest_to_pinecone.py

Or with environment variables inline:
    VECTOR_STORE_TYPE=pinecone PINECONE_API_KEY=xxx python scripts/ingest_to_pinecone.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Force Pinecone mode
os.environ["VECTOR_STORE_TYPE"] = "pinecone"

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from rag_engine import TaxRagEngine


def main():
    # Validate environment
    if not os.getenv("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY environment variable is required.")
        print("Get your API key from https://www.pinecone.io")
        sys.exit(1)

    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable is required.")
        print("Get your API key from https://aistudio.google.com/apikey")
        sys.exit(1)

    index_name = os.getenv("PINECONE_INDEX_NAME", "ato-tax-assistant")
    print(f"Pinecone index: {index_name}")
    print(f"Vector store type: {os.getenv('VECTOR_STORE_TYPE')}")

    # Initialize engine
    print("\nInitializing TaxRagEngine...")
    engine = TaxRagEngine(model_provider="google")

    # Ingest PDF
    pdf_path = project_root / "data" / "tax_guide.pdf"
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        sys.exit(1)

    print(f"\nIngesting {pdf_path} to Pinecone...")
    num_chunks = engine.ingest_pdf(str(pdf_path))
    print(f"\nSuccess! Ingested {num_chunks} chunks to Pinecone index '{index_name}'")

    # Test query
    print("\nTesting query...")
    result = engine.ask("What is the tax-free threshold?")
    print(f"Answer: {result.answer[:200]}...")
    print(f"Sources: {len(result.source_documents)} documents")

    print("\nDone! Your Pinecone index is ready for Streamlit Cloud deployment.")


if __name__ == "__main__":
    main()
