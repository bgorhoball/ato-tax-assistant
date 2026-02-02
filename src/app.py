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
    st.text(f"Google API Key: {google_key}")
    st.text(f"OpenAI API Key: {openai_key}")

    # Check vector store
    chroma_path = Path(__file__).parent.parent / "chroma_db"
    vectorstore_status = "✅" if chroma_path.exists() else "❌"
    st.text(f"Vector Store: {vectorstore_status}")

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

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 Source Documents"):
                for i, doc in enumerate(message["sources"], 1):
                    page = doc.metadata.get("page", "N/A")
                    st.markdown(f"**Source {i}** (Page {page})")
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

            # Show source documents
            if result.source_documents:
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result.source_documents, 1):
                        page = doc.metadata.get("page", "N/A")
                        st.markdown(f"**Source {i}** (Page {page})")
                        st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                        st.divider()

            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.answer,
                "sources": result.source_documents,
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
