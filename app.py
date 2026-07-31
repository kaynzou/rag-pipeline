"""
Streamlit frontend for the RAG pipeline.

Connects to the FastAPI server and provides a chat interface
for asking questions and seeing cited answers.
"""

import requests
import streamlit as st

st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="",
    layout="wide",
)

st.title("RAG Pipeline — Chat with Your Documents")
st.markdown(
    "Ask questions about the indexed corpus. "
    "Answers are grounded in retrieved context with source citations."
)

API_URL = "http://localhost:8000/query"
HEALTH_URL = "http://localhost:8000/ready"


@st.cache_data(ttl=5)
def check_health():
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        return r.json().get("ready", False)
    except Exception:
        return False


with st.sidebar:
    st.header("Status")
    ready = check_health()
    if ready:
        st.success("Pipeline ready")
    else:
        st.error("Pipeline not ready — start the FastAPI server first")

    st.header("Settings")
    top_k = st.slider("Top-k results", 1, 10, 5)

if ready:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"**chunk_id={s['chunk_id']}** "
                            f"(score={s['rerank_score']:.3f}) — {s['source_file']}"
                        )
                        st.caption(s["text_preview"])

    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        API_URL,
                        json={"question": prompt, "top_k": top_k},
                        timeout=60,
                    )
                    response.raise_for_status()
                    data = response.json()

                    st.markdown(data["answer"])

                    with st.expander("Sources"):
                        for s in data["sources"]:
                            st.markdown(
                                f"**chunk_id={s['chunk_id']}** "
                                f"(score={s['rerank_score']:.3f}) — {s['source_file']}"
                            )
                            st.caption(s["text_preview"])

                    st.caption(
                        f"Retrieved {data['chunks_retrieved']} chunks, "
                        f"used {data['chunks_used']} | "
                        f"Latency: {data['latency_ms']:.0f}ms"
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": data["answer"],
                            "sources": data["sources"],
                        }
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
else:
    st.info("Start the FastAPI server: `uvicorn src.server:app --reload`")
