import sys
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import streamlit.components.v1 as components

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

# ------------------------------------------------------------
# CLEAR-QUESTION FLAG
#
# The form runs with clear_on_submit=False, because
# clear_on_submit=True wipes EVERY widget in the form back to
# its default after each submit — including the category
# selectbox. The question box is cleared manually instead: we
# set this flag when a question is sent, then reset the text
# input's session_state value at the very top of the NEXT run,
# before the widget with that key is instantiated. That's the
# only thing that resets — the category keeps whatever the
# user picked.
# ------------------------------------------------------------

if "clear_question" not in st.session_state:
    st.session_state.clear_question = False

if st.session_state.clear_question:
    st.session_state.question_input = ""
    st.session_state.clear_question = False

# ------------------------------------------------------------
# PENDING QUESTION
#
# Sending is split into two steps so the question box empties
# IMMEDIATELY on submit, before the answer is computed — rather
# than sitting there full for the whole spinner and only
# clearing once the answer is already on screen.
#
# Step 1 (in the "if send:" block below): stash the question,
# mode, and category, set clear_question, and rerun right away.
# That rerun clears the box on the very next render.
#
# Step 2 (the "if st.session_state.pending_question:" block
# further down): on that same next run, the box is already
# empty, so we go ahead and actually process the stashed
# question — spinner, answer, save to chat_log.
# ------------------------------------------------------------

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "pending_mode" not in st.session_state:
    st.session_state.pending_mode = None

if "pending_category" not in st.session_state:
    st.session_state.pending_category = None


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
       MODE CAPTION
       Shown above the input row via st.caption(), not as a
       hand-positioned <div> squeezed into a narrow column next
       to the other widgets. Plain block-level text always
       renders at its natural width and can never overlap a
       neighboring widget, so this replaces the old
       .chat-search-label / .temporary-label / .chat-search
       divs entirely.
       ======================================================== */

    .st-key-fixed_chatbar div[data-testid="stCaptionContainer"] {

        padding: 0 4px 6px 4px;

        margin: 0;
    }


    /* ========================================================
       COLUMN CLIPPING
       Keeps any widget from drawing over its neighbor if a
       column ever ends up narrower than its content.
       ======================================================== */

    .st-key-fixed_chatbar
    div[data-testid="stHorizontalBlock"]
    div[data-testid="column"] {

        overflow: hidden;

        min-width: 0;
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
    #
    # Indexing runs in the BACKGROUND (subprocess.Popen instead
    # of subprocess.run) so the UI stays responsive — you can
    # keep asking questions while a big file finishes embedding
    # instead of the whole app freezing until it's done. A
    # fragment polls the process every 2s and reports progress.
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


    INGEST_LOG_PATH = Path(__file__).resolve().parents[1] / "vector_store" / "ingest_log.txt"

    if "ingest_process" not in st.session_state:
        st.session_state.ingest_process = None
    if "ingest_filename" not in st.session_state:
        st.session_state.ingest_filename = None

    if uploaded_file is not None:
        target_dir = RAW_DATA_DIR / upload_group
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / uploaded_file.name

        is_currently_being_indexed = (
            st.session_state.ingest_process is not None
            and st.session_state.ingest_filename == uploaded_file.name
        )

        if is_currently_being_indexed:
            pass  # don't show anything here — the progress block below already covers it
        elif target_path.exists():
            st.warning(f"⚠️ '{uploaded_file.name}' already exists in **{upload_group}**. It's already part of the knowledge base — no need to upload it again.")
        elif st.session_state.ingest_process is not None:
            st.info(f"⏳ Still indexing '{st.session_state.ingest_filename}' in the background. Please wait before uploading another file.")
        else:
            if st.button("Upload and Index", key="upload_index_button"):
                with open(target_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                INGEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

                proc = subprocess.Popen(
                    [sys.executable, "-u", "-m", "src.ingest.embed_and_store"],
                    cwd=str(Path(__file__).resolve().parents[1])
                )

                st.session_state.ingest_process = proc
                st.session_state.ingest_filename = uploaded_file.name

                st.info(
                    f"Started indexing '{uploaded_file.name}' in the background. "
                    f"You can keep asking questions — this will finish on its own."
                )

                st.rerun()

    # --- Check on background indexing every rerun (non-blocking) ---
    # --- Automatically check background indexing ---
    @st.fragment(run_every="2s")
    def check_indexing_status():

        if st.session_state.ingest_process is None:
            return

        proc = st.session_state.ingest_process
        return_code = proc.poll()

        if return_code is None:

            st.info(
                f"⏳ Indexing "
                f"'{st.session_state.ingest_filename}' "
                f"in progress..."
            )

        elif return_code == 0:

            filename = st.session_state.ingest_filename

            st.success(
                f"✅ '{filename}' indexed successfully!"
            )

            # Reload the newly updated index
            load_index()

            # Clear process state
            st.session_state.ingest_process = None
            st.session_state.ingest_filename = None

        else:

            filename = st.session_state.ingest_filename

            st.error(
                f"❌ Indexing '{filename}' failed."
            )

            if INGEST_LOG_PATH.exists():
                st.code(
                    INGEST_LOG_PATH.read_text()[-2000:]
                )

            st.session_state.ingest_process = None
            st.session_state.ingest_filename = None


    check_indexing_status()


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
        clear_on_submit=False,
        border=False
    ):

        # ----------------------------------------------------
        # MODE CAPTION
        #
        # Plain st.caption() above the row instead of a
        # hand-positioned label div squeezed into a narrow
        # column beside the other widgets. Renders at its own
        # natural width every time, so it can't overlap the
        # dropdown or the send button on any screen size.
        # ----------------------------------------------------

        if mode == "Knowledge Base":

            st.caption("🔎 Searching the knowledge base — select a category:")

        else:

            st.caption("📄 Answering from your temporary document")


        # ----------------------------------------------------
        # QUESTION | CATEGORY | SEND
        # ----------------------------------------------------

        col_question, col_category, col_send = st.columns(
            [7.5, 1.8, 0.7],
            vertical_alignment="center"
        )


        # ====================================================
        # QUESTION
        # ====================================================

        with col_question:

            question = st.text_input(
                "Question",
                placeholder="Describe your issue...",
                label_visibility="collapsed",
                key="question_input"
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


        # ====================================================
        # SEND BUTTON
        # ====================================================

        with col_send:

            send = st.form_submit_button(
                "↑",
                use_container_width=True
            )


# ============================================================
# HANDLE SEND: STASH + CLEAR IMMEDIATELY
# ============================================================

if send:

    if not question.strip():

        st.warning(
            "Please describe your issue before sending."
        )

    else:

        # ----------------------------------------------------
        # Stash everything the processing step needs, clear the
        # question box, and rerun right away — so the box is
        # already empty by the time the answer starts loading,
        # instead of staying full through the whole spinner.
        # ----------------------------------------------------

        st.session_state.pending_question = question.strip()
        st.session_state.pending_mode = mode
        st.session_state.pending_category = selected_category

        st.session_state.clear_question = True

        st.rerun()


# ============================================================
# PROCESS PENDING QUESTION
#
# Runs on the rerun right after send — the question box is
# already blank at this point (cleared above), so the answer
# now loads with an empty box visible, matching the original
# immediate-clear feel.
# ============================================================

if st.session_state.pending_question:

    pending_q = st.session_state.pending_question
    pending_mode = st.session_state.pending_mode
    pending_category = st.session_state.pending_category


    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    st.session_state.chat_log.append(
        {
            "role": "user",
            "content": pending_q
        }
    )


    # ========================================================
    # DISPLAY USER MESSAGE
    # ========================================================

    with st.chat_message("user"):

        st.markdown(pending_q)


    # ========================================================
    # TEMPORARY DOCUMENT MODE
    # ========================================================

    if pending_mode == "Temporary Document":

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
                    pending_q
                )


            answer_text = result["answer"]

            sources = []


        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            st.markdown(
                answer_text
            )


        # ----------------------------------------------------
        # SAVE ANSWER
        # ----------------------------------------------------

        st.session_state.chat_log.append(
            {
                "role": "assistant",
                "content": answer_text,
                "sources": sources
            }
        )


    # ========================================================
    # KNOWLEDGE BASE MODE
    # ========================================================

    else:

        # ----------------------------------------------------
        # CATEGORY FILTER
        # ----------------------------------------------------

        group_filter = (
            pending_category
            if pending_category != "All"
            else None
        )


        # ----------------------------------------------------
        # SEARCH KNOWLEDGE BASE
        # ----------------------------------------------------

        with st.spinner(
            "Searching knowledge base..."
        ):

            result, st.session_state.history = (
                answer_ticket(
                    pending_q,
                    st.session_state.history,
                    group_filter=group_filter
                )
            )


        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            st.markdown(
                result["answer"]
            )


            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            if result["sources"]:

                with st.expander(
                    "📚 Sources"
                ):

                    for source in result["sources"]:

                        st.write(
                            source["doc"]
                        )


        # ----------------------------------------------------
        # SAVE ANSWER
        # ----------------------------------------------------

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


    # ========================================================
    # CLEAR PENDING STATE, THEN RERUN TO SETTLE
    # ========================================================

    st.session_state.pending_question = None
    st.session_state.pending_mode = None
    st.session_state.pending_category = None

    st.rerun()


# ============================================================
# AUTO-SCROLL TO BOTTOM
#
# Streamlit doesn't scroll the page on its own when new content
# is added, so after a new answer lands the user is often left
# looking at the middle of the conversation instead of the
# latest message. This runs at the very end of every completed
# run (i.e. once everything, including a freshly answered
# question, is already in st.session_state.chat_log and drawn
# on screen) and scrolls the page down to the bottom.
#
# It has to go through components.html rather than
# st.markdown(..., unsafe_allow_html=True): Streamlit inserts
# markdown HTML directly into the page, where <script> tags are
# not executed. components.html renders inside its own iframe,
# where scripts DO run, so it reaches out to window.parent (the
# actual app page) to do the scrolling. height=0 keeps that
# iframe invisible.
# ============================================================

components.html(
    """
    <script>
        function scrollChatToBottom() {
            var doc = window.parent.document;

            // Streamlit wraps the page content in one of these,
            // depending on version, and THIS is what actually
            // scrolls (the outer window usually does not) — so
            // scrolling window.parent alone had no effect.
            var container = (
                doc.querySelector('[data-testid="stMain"]') ||
                doc.querySelector('section.main') ||
                doc.querySelector('[data-testid="stAppViewContainer"]')
            );

            if (container) {
                container.scrollTop = container.scrollHeight;
            }

            // Fallback, in case some Streamlit version scrolls
            // the window itself instead of an inner container.
            window.parent.scrollTo(0, doc.body.scrollHeight);
        }

        // Retried a few times: right after a rerun, Streamlit
        // may still be laying out the newest message, so the
        // scrollHeight read on the very first attempt can be
        // stale (too short) — later attempts catch it once
        // everything has actually rendered.
        setTimeout(scrollChatToBottom, 50);
        setTimeout(scrollChatToBottom, 200);
        setTimeout(scrollChatToBottom, 500);
    </script>
    """,
    height=0
)