import os
import sys
from pathlib import Path

import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag_engine import TaxRagEngine

# Page config
st.set_page_config(
    page_title="ATO Tax AI Assistant",
    page_icon="🇦🇺",
    layout="wide",
)

# Sidebar
with st.sidebar:
    st.header("Settings")

    # Model provider selection
    provider = st.radio(
        "Select AI Provider",
        options=["Google Gemini", "OpenAI"],
        index=0,
        help="Choose the AI model provider for answering questions",
    )
    provider_key = "google" if provider == "Google Gemini" else "openai"

    st.divider()

    # Show environment status
    st.subheader("Status")
    google_key = "✅" if os.getenv("GOOGLE_API_KEY") else "❌"
    openai_key = "✅" if os.getenv("OPENAI_API_KEY") else "❌"
    pinecone_key = "✅" if os.getenv("PINECONE_API_KEY") else "❌"
    st.text(f"Google API Key: {google_key}")
    st.text(f"OpenAI API Key: {openai_key}")
    st.text(f"Pinecone API Key: {pinecone_key}")

    # Show vector store type
    vector_store_type = os.getenv("VECTOR_STORE_TYPE", "chroma")
    if vector_store_type == "pinecone":
        store_icon = "☁️"
        store_name = "Pinecone (Cloud)"
    else:
        store_icon = "💾"
        store_name = "ChromaDB (Local)"
        # Check if local vector store exists
        chroma_path = Path(__file__).parent.parent / "chroma_db"
        if not chroma_path.exists():
            store_name += " ❌"

    st.text(f"Vector Store: {store_icon} {store_name}")

    st.divider()
    st.caption("Built with LangChain + Streamlit")


@st.cache_resource
def get_engine(provider: str) -> TaxRagEngine:
    """Initialize and cache the RAG engine."""
    return TaxRagEngine(
        model_provider=provider,
        persist_directory=str(Path(__file__).parent.parent / "chroma_db"),
    )


# Main content
st.title("🇦🇺 ATO Tax AI Assistant")
st.caption("Ask questions about Australian tax returns based on official ATO documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_confidence_display(scores: list) -> tuple[float, str, str]:
    """Convert distance scores to confidence display."""
    if not scores:
        return 0.0, "N/A", "⚪"

    # Chroma returns distance (lower = better), typically 0-2 for cosine
    top_score = min(scores)
    # Convert distance to similarity (0-1 scale)
    confidence = max(0, min(1, 1 - top_score / 2))

    if confidence >= 0.7:
        level, indicator = "High", "🟢"
    elif confidence >= 0.4:
        level, indicator = "Medium", "🟡"
    else:
        level, indicator = "Low", "🔴"

    return confidence, level, indicator


def render_confidence_dashboard(scores: list, docs: list):
    """Render the AI confidence analysis dashboard."""
    confidence, level, indicator = get_confidence_display(scores)

    st.markdown("### 🤖 AI Confidence Analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Top Match Similarity",
            value=f"{confidence:.2f}",
            help="Similarity score (0-1, higher is better)"
        )
    with col2:
        st.metric(
            label="Confidence Level",
            value=f"{level} {indicator}",
            help="Based on vector similarity threshold"
        )
    with col3:
        st.metric(
            label="Sources Retrieved",
            value=len(docs),
            help="Number of relevant document chunks"
        )

    # Show top match info
    if docs and scores:
        top_idx = scores.index(min(scores))
        top_doc = docs[top_idx]
        page = top_doc.metadata.get("page", "N/A")
        st.caption(
            f"📍 **Logic:** The vector of your question is most aligned with "
            f"**Page {page}** (distance: {min(scores):.4f})"
        )


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show confidence dashboard for assistant messages
        if message.get("scores"):
            with st.expander("🤖 AI Confidence Analysis", expanded=False):
                render_confidence_dashboard(message["scores"], message.get("sources", []))

        if message.get("sources"):
            with st.expander("📄 Source Documents"):
                scores = message.get("scores", [])
                for i, doc in enumerate(message["sources"], 1):
                    page = doc.metadata.get("page", "N/A")
                    score_text = f" | Distance: {scores[i-1]:.4f}" if i <= len(scores) else ""
                    st.markdown(f"**Source {i}** (Page {page}{score_text})")
                    st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about Australian tax..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        try:
            engine = get_engine(provider_key)

            with st.spinner("Thinking..."):
                result = engine.ask(prompt)

            st.markdown(result.answer)

            # Show confidence dashboard
            if result.scores:
                with st.expander("🤖 AI Confidence Analysis", expanded=True):
                    render_confidence_dashboard(result.scores, result.source_documents)

            # Show source documents
            if result.source_documents:
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result.source_documents, 1):
                        page = doc.metadata.get("page", "N/A")
                        score_text = f" | Distance: {result.scores[i-1]:.4f}" if i <= len(result.scores) else ""
                        st.markdown(f"**Source {i}** (Page {page}{score_text})")
                        st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                        st.divider()

            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.answer,
                "sources": result.source_documents,
                "scores": result.scores,
            })

        except FileNotFoundError as e:
            st.error(f"Vector store not found. Please run the ingestion script first:\n```\npython src/rag_engine.py\n```")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.divider()
col1, col2 = st.columns(2)
with col1:
    st.caption("Data source: ATO Individual Tax Return Instructions 2024")
with col2:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
