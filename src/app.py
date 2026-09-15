# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parents[1]))

# import subprocess
# import streamlit as st
# from src.agent.agent import answer_ticket


# # Page configuration
# st.set_page_config(
#     page_title="Troubleshooting Agent",
#     page_icon="🤖",
#     layout="wide"
# )


# # Title
# st.title("🤖Troubleshooting Agent")
# st.caption("Ask questions about your troubleshooting documentation")


# # --- Upload sidebar ---
# GROUPS = ["veeam", "sops", "tickets", "kb", "repository", "vm_virtualization"]
# RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# with st.sidebar:
#     st.header("📤 Upload Document")
#     selected_group = st.selectbox("Select category", GROUPS)
#     uploaded_file = st.file_uploader(
#         "Choose a file",
#         type=["pdf", "docx", "xlsx", "xls", "csv", "txt", "md"]
#     )

#     if uploaded_file is not None:
#         if st.button("Upload and Index"):
#             target_dir = RAW_DATA_DIR / selected_group
#             target_dir.mkdir(parents=True, exist_ok=True)
#             target_path = target_dir / uploaded_file.name

#             with open(target_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())

#             st.info(f"Saved {uploaded_file.name} to {selected_group}. Indexing now...")

#             with st.spinner("Embedding new document — this may take a moment..."):
#                 result = subprocess.run(
#                     [sys.executable, "-m", "src.ingest.embed_and_store"],
#                     capture_output=True,
#                     text=True,
#                     cwd=str(Path(__file__).resolve().parents[1])
#                 )

#             if result.returncode == 0:
#                 st.success(f"'{uploaded_file.name}' indexed successfully!")
#             else:
#                 st.error("Indexing failed. Check details below.")
#                 st.code(result.stderr[-2000:])


# # Store conversation history in Streamlit session
# if "history" not in st.session_state:
#     st.session_state.history = []


# # Display previous messages
# for message in st.session_state.history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])


# # Chat input
# CATEGORIES = ["All", "vmware", "netapp", "dell_emc", "veeam", "exagrid", "hitachi", "microsoft"]
# selected_category = st.selectbox("Search within category", CATEGORIES)
# question = st.chat_input("Describe your issue...")


# if question:

#     # Display user's question
#     with st.chat_message("user"):
#         st.markdown(question)

#     # Call your existing RAG agent
#     result, st.session_state.history = answer_ticket(
#     question,
#     st.session_state.history,
#     group_filter=selected_category if selected_category != "All" else None
# )

#     # Display agent response
#     with st.chat_message("assistant"):
#         st.markdown(result["answer"])

#         # Display sources
#         if result["sources"]:
#             with st.expander("📚 Sources"):
#                 for source in result["sources"]:
#                     st.write(source["doc"])




import sys
import subprocess
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
from src.agent.agent import answer_ticket
from src.temp_doc.temp_agent import answer_from_temp_doc


# Page configuration
st.set_page_config(
    page_title="Troubleshooting Agent",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Troubleshooting Agent")
st.caption("Ask questions about your troubleshooting documentation")


# --- Config ---
CATEGORIES = ["All", "vmware", "netapp", "dell_emc", "veeam", "exagrid", "hitachi", "microsoft"]
RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


# --- Sidebar: permanent knowledge base upload ---
with st.sidebar:
    st.header("📤 Add to Knowledge Base")
    st.caption("Permanently indexed — searchable by everyone, always.")

    upload_group = st.selectbox("Select category", CATEGORIES[1:])  # exclude "All" here
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "xlsx", "xls", "csv", "txt", "md"],
        key="permanent_uploader"
    )

    if uploaded_file is not None:
        if st.button("Upload and Index"):
            target_dir = RAW_DATA_DIR / upload_group
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / uploaded_file.name

            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.info(f"Saved {uploaded_file.name} to {upload_group}. Indexing now...")

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

    st.divider()

    # --- Sidebar: temporary document upload ---
    st.header("📄 Ask About a Temporary Document")
    st.caption("NOT saved — only answers questions for this session.")

    temp_file = st.file_uploader(
        "Upload a document (temporary)",
        type=["pdf", "docx", "xlsx", "xls", "csv", "txt", "md"],
        key="temp_uploader"
    )

    if temp_file is not None:
        st.session_state.temp_file = temp_file
        st.success(f"'{temp_file.name}' loaded for this session (not saved).")


# --- Main area: mode + category selection ---
col1, col2 = st.columns([2, 1])
with col1:
    mode = st.radio(
        "Ask questions using:",
        ["Knowledge Base", "Temporary Document"],
        horizontal=True
    )
with col2:
    if mode == "Knowledge Base":
        selected_category = st.selectbox("Search within category", CATEGORIES)
    else:
        selected_category = "All"


# --- Session state ---
if "history" not in st.session_state:
    st.session_state.history = []

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

if "temp_file" not in st.session_state:
    st.session_state.temp_file = None


# --- Display previous messages ---
for entry in st.session_state.chat_log:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry["role"] == "assistant" and entry.get("sources"):
            with st.expander("📚 Sources"):
                for s in entry["sources"]:
                    st.write(s)


# --- Chat input ---
question = st.chat_input("Describe your issue...")

if question:
    st.session_state.chat_log.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if mode == "Temporary Document":
        if st.session_state.temp_file is None:
            answer_text = "Please upload a temporary document first (see sidebar)."
            sources = []
        else:
            with st.spinner("Reading document and answering..."):
                result = answer_from_temp_doc(st.session_state.temp_file, question)
            answer_text = result["answer"]
            sources = []

        with st.chat_message("assistant"):
            st.markdown(answer_text)

        st.session_state.chat_log.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources
        })

    else:
        group_filter = selected_category if selected_category != "All" else None

        with st.spinner("Searching knowledge base..."):
            result, st.session_state.history = answer_ticket(
                question,
                st.session_state.history,
                group_filter=group_filter
            )

        with st.chat_message("assistant"):
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("📚 Sources"):
                    for source in result["sources"]:
                        st.write(source["doc"])

        st.session_state.chat_log.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": [s["doc"] for s in result["sources"]]
        })