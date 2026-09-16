import sys
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.agent.agent import answer_ticket
from src.temp_doc.temp_agent import answer_from_temp_doc
from src.retrieval.retriever import load_index


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Troubleshooting Agent",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

CATEGORIES = [
    "All",
    "vmware",
    "netapp",
    "dell_emc",
    "veeam",
    "exagrid",
    "hitachi",
    "microsoft"
]

RAW_DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

if "temp_file" not in st.session_state:
    st.session_state.temp_file = None

if "question_mode" not in st.session_state:
    st.session_state.question_mode = "Knowledge Base"

if "chat_category" not in st.session_state:
    st.session_state.chat_category = "All"


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       PAGE
       ======================================================== */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 125px;
    }


    /* ========================================================
       FIXED CHAT BAR
       ======================================================== */

    .st-key-fixed_chatbar {

        position: fixed;

        left: 21.5rem;
        right: 1.5rem;
        bottom: 18px;

        z-index: 999;

        background: #ffffff;

        border: 1px solid #d6d9df;

        border-radius: 17px;

        padding: 8px 10px;

        box-shadow:
            0 4px 18px rgba(0, 0, 0, 0.08);
    }


    /* ========================================================
       FORM
       ======================================================== */

    .st-key-fixed_chatbar [data-testid="stForm"] {

        border: none !important;

        padding: 0 !important;

        margin: 0 !important;

        background: transparent !important;
    }


    /* ========================================================
       REMOVE DEFAULT SPACING
       ======================================================== */

    .st-key-fixed_chatbar
    div[data-testid="stTextInput"] {

        margin: 0 !important;

        padding: 0 !important;
    }

    .st-key-fixed_chatbar
    div[data-testid="stSelectbox"] {

        margin: 0 !important;

        padding: 0 !important;
    }

    .st-key-fixed_chatbar
    div[data-testid="stFormSubmitButton"] {

        margin: 0 !important;

        padding: 0 !important;
    }


    /* ========================================================
       QUESTION INPUT
       ======================================================== */

    .st-key-fixed_chatbar
    div[data-testid="stTextInput"] input {

        height: 44px !important;

        min-height: 44px !important;

        box-sizing: border-box;

        border: 1px solid #e1e4e8 !important;

        border-radius: 10px !important;

        background: #f5f7fa !important;

        box-shadow: none !important;

        padding-left: 14px !important;

        padding-right: 14px !important;

        font-size: 16px !important;

        color: #202124 !important;
    }


    .st-key-fixed_chatbar
    div[data-testid="stTextInput"] input:focus {

        border: 1px solid #c7cbd1 !important;

        box-shadow: none !important;

        outline: none !important;
    }


    /* ========================================================
       SEARCH LABEL
       ======================================================== */

    .chat-search-label {

        height: 44px;

        display: flex;

        align-items: top;

        justify-content: flex-end;

        box-sizing: border-box;

        padding-right: 5px;
        padding-bottom: 60px;

        font-size: 20px;

        line-height: 44px;

        font-weight: 400;

        color: #4b5563;

        white-space: nowrap;

        transform: translateY(-1px);
    }


    /* ========================================================
       CATEGORY SELECTBOX
       ======================================================== */

    .st-key-fixed_chatbar
    div[data-testid="stSelectbox"] > div {

        margin: 0 !important;

        padding: 0 !important;
    }


    .st-key-fixed_chatbar
    div[data-testid="stSelectbox"] > div > div {

        height: 44px !important;

        min-height: 44px !important;

        box-sizing: border-box;

        border: 1px solid #e1e4e8 !important;

        border-radius: 10px !important;

        background: #f5f7fa !important;

        box-shadow: none !important;

        font-size: 16px !important;
    }


    /* Baseweb select */
    .st-key-fixed_chatbar
    div[data-testid="stSelectbox"]
    [data-baseweb="select"] {

        height: 44px !important;

        min-height: 44px !important;
    }


    .st-key-fixed_chatbar
    div[data-testid="stSelectbox"]
    [data-baseweb="select"] > div {

        height: 44px !important;

        min-height: 44px !important;

        display: flex !important;

        align-items: center !important;

        box-sizing: border-box;
    }


    /* Selectbox text */
    .st-key-fixed_chatbar
    div[data-testid="stSelectbox"]
    [data-baseweb="select"] span {

        font-size: 16px !important;

        color: #374151 !important;
    }


    /* ========================================================
       SEND BUTTON
       ======================================================== */

    .st-key-fixed_chatbar
    div[data-testid="stFormSubmitButton"] button {

        width: 44px !important;

        height: 44px !important;

        min-width: 44px !important;

        min-height: 44px !important;

        padding: 0 !important;

        margin: 0 !important;

        border-radius: 50% !important;

        border: none !important;

        background: #111827 !important;

        color: white !important;

        font-size: 21px !important;

        display: flex !important;

        align-items: center !important;

        justify-content: center !important;

        line-height: 1 !important;
    }


    .st-key-fixed_chatbar
    div[data-testid="stFormSubmitButton"] button:hover {

        background: #374151 !important;

        color: white !important;
    }


    /* ========================================================
       TEMPORARY DOCUMENT
       ======================================================== */

    .temporary-label {

        height: 44px;

        display: flex;

        align-items: center;

        justify-content: center;

        font-size: 16px;

        color: #6b7280;

        white-space: nowrap;
    }


    /* ========================================================
       CHAT MESSAGES
       ======================================================== */

    [data-testid="stChatMessage"] {

        padding-top: 0.5rem;

        padding-bottom: 0.5rem;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 900px) {

        .st-key-fixed_chatbar {

            left: 1rem;

            right: 1rem;

            bottom: 10px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Troubleshooting Agent")

st.caption(
    "Ask questions about your troubleshooting documentation"
)


# --- Config ---
CATEGORIES = ["All", "vmware", "netapp", "dell_emc", "veeam", "exagrid", "hitachi", "microsoft", "reference"]
RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


# --- Sidebar: permanent knowledge base upload ---
# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # ========================================================
    # QUESTION SETTINGS
    # ========================================================

    st.header("⚙️ Question Settings")

    mode = st.radio(
        "Ask questions using:",
        [
            "Knowledge Base",
            "Temporary Document"
        ],
        key="question_mode"
    )


    st.divider()


    # ========================================================
    # ADD TO KNOWLEDGE BASE
    # ========================================================

    st.header("📤 Add to Knowledge Base")

    st.caption(
        "Permanently indexed — searchable by everyone, always."
    )


    upload_group = st.selectbox(
        "Select category",
        CATEGORIES[1:],
        key="upload_group"
    )


    uploaded_file = st.file_uploader(
        "Choose a file",
        type=[
            "pdf",
            "docx",
            "xlsx",
            "xls",
            "csv",
            "txt",
            "md"
        ],
        key="permanent_uploader"
    )


    if uploaded_file is not None:

        if st.button(
            "Upload and Index",
            key="upload_index_button"
        ):

            target_dir = RAW_DATA_DIR / upload_group

            target_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            target_path = (
                target_dir
                / uploaded_file.name
            )


            with open(target_path, "wb") as f:

                f.write(
                    uploaded_file.getbuffer()
                )


            st.info(
                f"Saved {uploaded_file.name} "
                f"to {upload_group}. Indexing now..."
            )


            with st.spinner(
                "Embedding new document — "
                "this may take a moment..."
            ):

                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "src.ingest.embed_and_store"
                    ],
                    capture_output=True,
                    text=True,
                    cwd=str(
                        Path(__file__).resolve().parents[1]
                    )
                )


            if result.returncode == 0:

                st.success(
                    f"'{uploaded_file.name}' "
                    "indexed successfully!"
                )

                # Refresh in-memory index so the new
                # document is searchable immediately
                # (restored from original logic)
                load_index()

            else:

                st.error(
                    "Indexing failed. "
                    "Check details below."
                )

                st.code(
                    result.stderr[-2000:]
                )


    st.divider()


    # ========================================================
    # TEMPORARY DOCUMENT
    # ========================================================

    st.header(
        "📄 Ask About a Temporary Document"
    )

    st.caption(
        "NOT saved — only answers questions "
        "for this session."
    )


    temp_file = st.file_uploader(
        "Upload a document (temporary)",
        type=[
            "pdf",
            "docx",
            "xlsx",
            "xls",
            "csv",
            "txt",
            "md"
        ],
        key="temp_uploader"
    )


    if temp_file is not None:

        st.session_state.temp_file = temp_file

        st.success(
            f"'{temp_file.name}' loaded for "
            "this session (not saved)."
        )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for entry in st.session_state.chat_log:

    with st.chat_message(entry["role"]):

        st.markdown(
            entry["content"]
        )


        if (
            entry["role"] == "assistant"
            and entry.get("sources")
        ):

            with st.expander("📚 Sources"):

                for source in entry["sources"]:

                    st.write(source)


# ============================================================
# FIXED CHAT INPUT
# ============================================================

with st.container(
    key="fixed_chatbar"
):

    # ========================================================
    # FORM
    #
    # ENTER = SEND
    # BUTTON = SEND
    # ========================================================

    with st.form(
        key="chat_form",
        clear_on_submit=True,
        border=False
    ):

        # ----------------------------------------------------
        # QUESTION | SEARCH | CATEGORY | SEND
        # ----------------------------------------------------

        col_question, col_search, col_category, col_send = st.columns(
            [6.5, 0.9, 1.8, 0.6],
            vertical_alignment="center"
        )


        # ====================================================
        # QUESTION
        # ====================================================

        with col_question:

            question = st.text_input(
                "Question",
                placeholder="Describe your issue...",
                label_visibility="collapsed"
            )


        # ====================================================
        # SEARCH
        # ====================================================

        with col_search:

            if mode == "Knowledge Base":

                st.markdown(
                    """
                    <div class="chat-search-label">
                        Select:
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="chat-search-label">
                        Document:
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # ====================================================
        # CATEGORY
        # ====================================================

        with col_category:

            if mode == "Knowledge Base":

                selected_category = st.selectbox(
                    "Category",
                    CATEGORIES,
                    label_visibility="collapsed",
                    key="chat_category"
                )

            else:

                selected_category = "All"

                st.markdown(
                    """
                    <div class="temporary-label">
                        Temporary
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # ====================================================
        # SEND BUTTON
        # ====================================================

        with col_send:

            send = st.form_submit_button(
                "↑",
                use_container_width=True
            )


# ============================================================
# PROCESS QUESTION
# ============================================================

if send:

    # ========================================================
    # EMPTY QUESTION
    # ========================================================

    if not question.strip():

        st.warning(
            "Please describe your issue before sending."
        )


    else:

        question = question.strip()


        # ====================================================
        # SAVE USER MESSAGE
        # ====================================================

        st.session_state.chat_log.append(
            {
                "role": "user",
                "content": question
            }
        )


        # ====================================================
        # DISPLAY USER MESSAGE
        # ====================================================

        with st.chat_message("user"):

            st.markdown(question)


        # ====================================================
        # TEMPORARY DOCUMENT MODE
        # ====================================================

        if mode == "Temporary Document":

            if st.session_state.temp_file is None:

                answer_text = (
                    "Please upload a temporary document "
                    "first (see sidebar)."
                )

                sources = []


            else:

                with st.spinner(
                    "Reading document and answering..."
                ):

                    result = answer_from_temp_doc(
                        st.session_state.temp_file,
                        question
                    )


                answer_text = result["answer"]

                sources = []


            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            with st.chat_message("assistant"):

                st.markdown(
                    answer_text
                )


            # ------------------------------------------------
            # SAVE ANSWER
            # ------------------------------------------------

            st.session_state.chat_log.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources
                }
            )


        # ====================================================
        # KNOWLEDGE BASE MODE
        # ====================================================

        else:

            # ------------------------------------------------
            # CATEGORY FILTER
            # ------------------------------------------------

            group_filter = (
                selected_category
                if selected_category != "All"
                else None
            )


            # ------------------------------------------------
            # SEARCH KNOWLEDGE BASE
            # ------------------------------------------------

            with st.spinner(
                "Searching knowledge base..."
            ):

                result, st.session_state.history = (
                    answer_ticket(
                        question,
                        st.session_state.history,
                        group_filter=group_filter
                    )
                )


            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            with st.chat_message("assistant"):

                st.markdown(
                    result["answer"]
                )


                # --------------------------------------------
                # SOURCES
                # --------------------------------------------

                if result["sources"]:

                    with st.expander(
                        "📚 Sources"
                    ):

                        for source in result["sources"]:

                            st.write(
                                source["doc"]
                            )


            # ------------------------------------------------
            # SAVE ANSWER
            # ------------------------------------------------

            st.session_state.chat_log.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": [
                        source["doc"]
                        for source in result["sources"]
                    ]
                }
            )


        # ====================================================
        # RERUN
        # ====================================================

        st.rerun()