import os
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

API_URL = os.environ.get("API_URL", "http://localhost:8000/query")
HEALTH_URL = os.environ.get("HEALTH_URL", "http://localhost:8000/ready")
INDEX_URL = f"{API_URL.replace('/query', '')}/index"


@st.cache_data(ttl=5)
def check_health():
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        return r.json().get("ready", False)
    except Exception:
        return False


@st.cache_data(ttl=30)
def ensure_indexed():
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        if r.json().get("indexed"):
            return True
        requests.post(INDEX_URL, json={"source": "sample_corpus"}, timeout=120)
        return True
    except Exception:
        return False


with st.sidebar:
    st.header("Status")
    ready = check_health()
    if ready:
        st.success("Pipeline ready")
    else:
        st.error("Pipeline not ready — attempting to index...")
        if st.button("Retry"):
            st.cache_data.clear()
        st.stop()

    st.header("Settings")
    top_k = st.slider("Top-k results", 1, 10, 5)

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
