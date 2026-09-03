import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
from src.agent.agent import answer_ticket


# Page configuration
st.set_page_config(
    page_title="RAG Troubleshooting Agent",
    page_icon="🤖",
    layout="wide"
)


# Title
st.title("🤖 RAG Troubleshooting Agent")
st.caption("Ask questions about your troubleshooting documentation")


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