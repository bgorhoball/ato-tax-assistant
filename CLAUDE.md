# CLAUDE.md — ATO Tax Assistant

This file provides guidance for AI assistants working in this repository.

## Project Overview

The ATO Tax Assistant is a Retrieval-Augmented Generation (RAG) application that answers questions about Australian Tax Office (ATO) documents. Users upload PDF tax guides, which are chunked and stored in a vector database, then queried through a Streamlit chat interface backed by Google Gemini or OpenAI.

## Repository Structure

```
ato-tax-assistant/
├── src/
│   ├── app.py              # Streamlit web UI (chat interface, sidebar, confidence dashboard)
│   ├── rag_engine.py       # Synchronous RAG engine (TaxRagEngine)
│   ├── rag_engine_v2.py    # Async RAG engine (TaxRagEngineV2) — preferred for production
│   └── __init__.py
├── scripts/
│   ├── ingest_to_pinecone.py   # One-time sync PDF ingestion to Pinecone
│   └── aingest_to_pinecone.py  # Async PDF ingestion with CLI flags
├── data/                   # PDF files go here (gitignored except .gitkeep)
├── .devcontainer/
│   └── devcontainer.json   # VS Code Dev Container config (Python 3.11)
├── .env.example            # Required environment variable template
├── requirements.txt        # Python dependencies
└── TEST_SUMMARY.md         # Test results and sample Q&A outcomes
```

## Environment Setup

Copy `.env.example` to `.env` and populate:

```bash
VECTOR_STORE_TYPE=chroma          # "chroma" (local) or "pinecone" (cloud)
GOOGLE_API_KEY=...                 # Required for Gemini LLM + embeddings
OPENAI_API_KEY=...                 # Optional — only if using OpenAI provider
PINECONE_API_KEY=...               # Required only for cloud/production deployment
PINECONE_INDEX_NAME=ato-tax-assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run src/app.py
# Available at http://localhost:8501
```

The Dev Container auto-starts the server on attach with CORS and XSRF disabled for Codespaces compatibility.

## Ingesting PDFs

**Local (ChromaDB) — sync:**
```python
from src.rag_engine import TaxRagEngine
engine = TaxRagEngine()
engine.ingest_pdf("data/your-document.pdf")
```

**Cloud (Pinecone) — async (preferred):**
```bash
python scripts/aingest_to_pinecone.py --pdf data/your-document.pdf
python scripts/aingest_to_pinecone.py --compare  # benchmark sync vs async
```

## Architecture

```
Streamlit UI (src/app.py)
    └── TaxRagEngine / TaxRagEngineV2
            ├── LLM Provider: Google Gemini (gemini-2.5-flash) or OpenAI (gpt-4o)
            ├── Embeddings: Google (models/gemini-embedding-001, 768-dim) or OpenAI (text-embedding-3-small)
            └── Vector Store: ChromaDB (local, ./chroma_db/) or Pinecone (cloud)
```

### RAG Pipeline

1. PDF → `PyPDFLoader` → `RecursiveCharacterTextSplitter` (chunk: 1000 chars, overlap: 100)
2. Chunks → embedding model → vector store
3. User query → similarity search (k=4) → top chunks as context
4. Context + query → LLM with system prompt → answer + source citations

### Key Classes and Data Types

- `TaxRagEngine` (`rag_engine.py`): Synchronous. Use for simple scripts or local dev.
- `TaxRagEngineV2` (`rag_engine_v2.py`): Async-first. Preferred for production. Has sync wrappers for drop-in replacement.
- `AskResult`: Dataclass returned by `ask()` / `aask()` — fields: `answer`, `source_documents`, `similarity_scores`
- `IngestResult` (v2 only): Ingestion stats — fields: `total_chunks`, `successful`, `failed`, `duration_seconds`
- `RetryConfig` (v2 only): Tenacity retry parameters — `max_attempts=7`, backoff `5s→120s`

## Key Conventions

### LLM Configuration
- Temperature: `0.3` for both providers (controlled, factual responses)
- Google model: `gemini-2.5-flash`
- OpenAI model: `gpt-4o`
- Embedding dimension: `768` (Google) — Pinecone index must match

### Vector Store
- ChromaDB collection name: `ato_tax_documents`
- Pinecone index name: configurable via `PINECONE_INDEX_NAME`, default `ato-tax-assistant`
- `chroma_db/` is gitignored — regenerate locally from PDFs
- PDF files in `data/` are gitignored — provide your own source documents

### Rate Limiting and Retry (v2)
- Google free tier: 100 requests/min
- `TaxRagEngineV2` uses tenacity with exponential backoff (5s → 10s → 20s → 40s → 80s → 120s, max 7 attempts)
- SDK-level retries are disabled; tenacity handles all retry logic
- Default concurrency: `max_concurrency=3` (semaphore-controlled asyncio batches)

### Streamlit State Management
- RAG engine is cached with `@st.cache_resource` — one instance per session
- Chat history stored in `st.session_state.messages`
- Provider change triggers cache clear and engine re-initialization

## Development Workflow

### No test runner is configured. To validate changes:

1. Run the app: `streamlit run src/app.py`
2. Ingest a PDF and run sample queries
3. Review `TEST_SUMMARY.md` for expected behavior and tested configurations

### When modifying the RAG engine:
- Changes to `rag_engine.py` should be mirrored or superseded in `rag_engine_v2.py`
- The v2 engine has backwards-compatible sync methods (`ingest_pdf`, `ask`) — keep these working
- Embedding dimensions must match the Pinecone index (768 for Google, varies for OpenAI)
- Avoid changing chunk size/overlap without re-ingesting all documents

### When modifying `src/app.py`:
- Streamlit re-runs the entire script on each interaction — avoid side effects at module level
- Use `st.cache_resource` for expensive initializations (engine, connections)
- Confidence score UI reads from `AskResult.similarity_scores` — keep this field populated

### Adding a new LLM provider:
1. Add initialization logic in `_init_llm()` and `_init_embeddings()` in both engine files
2. Add the provider option to the sidebar in `src/app.py`
3. Update `.env.example` with required API key variables
4. Verify embedding dimension compatibility with configured vector store

### Adding a new vector store:
1. Implement `_ingest_to_<store>()` and loading logic in both engine files
2. Add `VECTOR_STORE_TYPE=<store>` to `.env.example`
3. Update `load_vectorstore()` dispatch in both engines

## Deployment

**Local:** ChromaDB (`VECTOR_STORE_TYPE=chroma`), run Streamlit directly.

**Streamlit Cloud / production:**
1. Set `VECTOR_STORE_TYPE=pinecone`
2. Ingest PDFs once using `scripts/aingest_to_pinecone.py`
3. Set all required env vars in the deployment environment
4. Deploy `src/app.py` as the entry point

The Dev Container (`postAttachCommand`) auto-starts the app — suitable for GitHub Codespaces demos.

## Important Constraints

- **No CI/CD pipeline** exists — all testing is manual
- **No type checking or linting** toolchain configured — follow existing code style
- **PDFs must be ingested before querying** — the app shows an error if the vector store is empty
- **Google free tier limits** apply during development — use async ingestion with retry for large PDFs
- **Pinecone index dimension must be 768** when using Google embeddings — mismatch causes ingestion failure
