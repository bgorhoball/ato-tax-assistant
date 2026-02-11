#!/usr/bin/env python3
"""
Async ingestion script to upload PDF documents to Pinecone.

This script uses the async TaxRagEngineV2 for faster ingestion
with concurrent batch processing.

Before running:
1. Create a free Pinecone account at https://www.pinecone.io
2. Create an index named 'ato-tax-assistant' with dimension 768
3. Set environment variables in .env:
   - PINECONE_API_KEY=your-api-key
   - PINECONE_INDEX_NAME=ato-tax-assistant
   - GOOGLE_API_KEY=your-google-api-key
   - VECTOR_STORE_TYPE=pinecone

Usage:
    python scripts/aingest_to_pinecone.py

Compare with sync version:
    python scripts/aingest_to_pinecone.py --compare
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Force Pinecone mode
os.environ["VECTOR_STORE_TYPE"] = "pinecone"

from dotenv import load_dotenv
load_dotenv(project_root / ".env")


def validate_environment():
    """Validate required environment variables."""
    if not os.getenv("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY environment variable is required.")
        print("Get your API key from https://www.pinecone.io")
        sys.exit(1)

    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable is required.")
        print("Get your API key from https://aistudio.google.com/apikey")
        sys.exit(1)


async def run_async_ingestion(pdf_path: Path) -> float:
    """Run async ingestion and return elapsed time."""
    from rag_engine_v2 import TaxRagEngineV2, RetryConfig

    print("\n" + "=" * 60)
    print("ASYNC INGESTION (TaxRagEngineV2)")
    print("=" * 60 + "\n")

    engine = TaxRagEngineV2(model_provider="google")

    # Configure retry with exponential backoff for rate limits
    # Google free tier: 100 requests/min, so we need longer waits
    retry_config = RetryConfig(
        max_attempts=7,
        min_wait=5.0,   # Start with 5s wait
        max_wait=120.0,  # Max 2 min wait for rate limit recovery
        exp_base=2.0,
    )

    result = await engine.aingest_pdf(
        str(pdf_path),
        batch_size=50,
        max_concurrency=3,
        retry_config=retry_config,
    )

    print(f"\nAsync Result:")
    print(f"  Total chunks: {result.total_chunks}")
    print(f"  Batches processed: {result.batches_processed}")
    print(f"  Elapsed time: {result.elapsed_time:.2f}s")

    # Test query
    print("\nTesting async query...")
    answer = await engine.aask("What is the tax-free threshold?")
    print(f"Answer: {answer.answer[:200]}...")

    return result.elapsed_time


def run_sync_ingestion(pdf_path: Path) -> float:
    """Run sync ingestion and return elapsed time."""
    from rag_engine import TaxRagEngine

    print("\n" + "=" * 60)
    print("SYNC INGESTION (TaxRagEngine)")
    print("=" * 60 + "\n")

    engine = TaxRagEngine(model_provider="google")

    start_time = time.time()
    num_chunks = engine.ingest_pdf(str(pdf_path))
    elapsed_time = time.time() - start_time

    print(f"\nSync Result:")
    print(f"  Total chunks: {num_chunks}")
    print(f"  Elapsed time: {elapsed_time:.2f}s")

    return elapsed_time


async def main():
    parser = argparse.ArgumentParser(description="Async PDF ingestion to Pinecone")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare async vs sync performance",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default="data/tax_guide.pdf",
        help="Path to PDF file",
    )
    args = parser.parse_args()

    validate_environment()

    index_name = os.getenv("PINECONE_INDEX_NAME", "ato-tax-assistant")
    print(f"Pinecone index: {index_name}")

    pdf_path = project_root / args.pdf
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        sys.exit(1)

    if args.compare:
        # Run both and compare
        print("\n" + "=" * 60)
        print("PERFORMANCE COMPARISON: ASYNC vs SYNC")
        print("=" * 60)

        # Note: For fair comparison, you'd need separate Pinecone indexes
        # or clear the index between runs. This just shows timing.
        async_time = await run_async_ingestion(pdf_path)

        print("\n" + "-" * 60)
        print("Note: Sync test skipped to avoid duplicate ingestion.")
        print("For accurate comparison, use separate indexes or clear between runs.")
        print("-" * 60)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Async time: {async_time:.2f}s")
        print(f"  Expected improvement: 2-3x faster with concurrent batches")

    else:
        # Just run async
        await run_async_ingestion(pdf_path)

    print("\nDone! Your Pinecone index is ready for Streamlit Cloud deployment.")


if __name__ == "__main__":
    asyncio.run(main())
