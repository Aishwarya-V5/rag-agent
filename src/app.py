import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import subprocess
import streamlit as st
from src.agent.agent import answer_ticket


# Page configuration
st.set_page_config(
    page_title="Troubleshooting Agent",
    page_icon="🤖",
    layout="wide"
)


# Title
st.title("🤖Troubleshooting Agent")
st.caption("Ask questions about your troubleshooting documentation")


# --- Upload sidebar ---
GROUPS = ["veeam", "sops", "tickets", "kb", "repository", "vm_virtualization"]
RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

with st.sidebar:
    st.header("📤 Upload Document")
    selected_group = st.selectbox("Select category", GROUPS)
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "xlsx", "xls", "csv", "txt", "md"]
    )

    if uploaded_file is not None:
        if st.button("Upload and Index"):
            target_dir = RAW_DATA_DIR / selected_group
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / uploaded_file.name

            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.info(f"Saved {uploaded_file.name} to {selected_group}. Indexing now...")

            with st.spinner("Embedding new document — this may take a moment..."):
                result = subprocess.run(
                    [sys.executable, "-m", "src.ingest.embed_and_store"],
                    capture_output=True,
                    text=True,
                    cwd=str(Path(__file__).resolve().parents[1])
                )

            if result.returncode == 0:
                st.success(f"'{uploaded_file.name}' indexed successfully!")
            else:
                st.error("Indexing failed. Check details below.")
                st.code(result.stderr[-2000:])


# Store conversation history in Streamlit session
if "history" not in st.session_state:
    st.session_state.history = []


# Display previous messages
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
question = st.chat_input("Describe your issue...")


if question:

    # Display user's question
    with st.chat_message("user"):
        st.markdown(question)

    # Call your existing RAG agent
    result, st.session_state.history = answer_ticket(
        question,
        st.session_state.history
    )

    # Display agent response
    with st.chat_message("assistant"):
        st.markdown(result["answer"])

        # Display sources
        if result["sources"]:
            with st.expander("📚 Sources"):
                for source in result["sources"]:
                    st.write(source["doc"])