import streamlit as st
import sys
from pathlib import Path

# Allow imports from src folder
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from rag_pipeline import (
    build_index_and_embeddings,
    retrieve_context,
    generate_answer
)
from ingest import prepare_chunks

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="chatINDECIMAL – AI Assistant",
    page_icon="🤖",
    layout="wide"
)
st.markdown("""
<style>
.chat-bubble-user {
    background-color: #DCF8C6;
    color: #000000;
    padding: 12px 16px;
    border-radius: 14px;
    margin: 10px 0 10px auto;
    width: fit-content;
    max-width: 75%;
    font-size: 16px;
}

.chat-bubble-bot {
    background-color: #2B2B2B;
    color: #FFFFFF;
    padding: 12px 16px;
    border-radius: 14px;
    margin: 10px auto 10px 0;
    width: fit-content;
    max-width: 75%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)


st.title("🤖 chatINDECIMAL")
st.caption("Document-grounded AI assistant (RAG) – no hallucinations")

# ------------------ INITIALIZE RAG ------------------
@st.cache_resource
def initialize_rag():
    chunks = prepare_chunks()
    model, index = build_index_and_embeddings(chunks)
    return chunks, model, index

chunks, model, index = initialize_rag()

# ------------------ SESSION STATE ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ CHAT INPUT ------------------
user_input = st.chat_input("Ask something about Indecimal…")

if user_input:
    # Save user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Retrieve & generate
    with st.spinner("🤖 Thinking..."):
        retrieved_chunks = retrieve_context(user_input, model, index, chunks)
        bot_answer = generate_answer(user_input, retrieved_chunks)

    # Save bot message
    st.session_state.chat_history.append({"role": "bot", "content": bot_answer})

# ------------------ CHAT DISPLAY ------------------
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"<div class='chat-bubble-user'>🧑 {message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-bot'>🤖 {message['content']}</div>", unsafe_allow_html=True)

# ------------------ CONTEXT VIEW ------------------
with st.expander("📄 View retrieved context (for transparency)"):
    if user_input:
        for i, chunk in enumerate(retrieved_chunks, 1):
            st.markdown(f"**Chunk {i} — {chunk['source']}**")
            st.markdown(
                f"<div style='font-size:14px; line-height:1.5;'>"
                f"{chunk['content'][:800]}..."
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("Ask a question to see retrieved context.")
