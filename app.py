
"""
Streamlit frontend for the RAG pipeline.

Connects to the FastAPI server and provides a chat interface
for asking questions and seeing cited answers. Supports voice input.
"""

import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Voice RAG Pipeline",
    page_icon="🎙️",
    layout="wide",
)

st.title("Voice-Enabled RAG Pipeline")
st.markdown(
    "Ask questions by typing or recording voice. "
    "Answers are grounded in retrieved context with source citations."
)

API_URL = os.environ.get("API_URL", "http://localhost:8000")
HEALTH_URL = f"{API_URL}/health"
READY_URL = f"{API_URL}/ready"
QUERY_URL = f"{API_URL}/query"
VOICE_URL = f"{API_URL}/voice/query"
LATENCY_URL = f"{API_URL}/latency"
INDEX_URL = f"{API_URL}/index"


@st.cache_data(ttl=5)
def check_health():
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        return r.json().get("status") == "ok"
    except Exception:
        return False


@st.cache_data(ttl=5)
def check_ready():
    try:
        r = requests.get(READY_URL, timeout=2)
        return r.json().get("ready", False)
    except Exception:
        return False


@st.cache_data(ttl=10)
def get_latency():
    try:
        r = requests.get(LATENCY_URL, timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


with st.sidebar:
    st.header("Status")
    ready = check_ready()
    if ready:
        st.success("Pipeline ready")
    else:
        st.error("Pipeline not ready — attempting to index...")
        if st.button("Retry"):
            st.cache_data.clear()
        st.stop()

    st.header("Latency Analytics")
    latency_data = get_latency()
    if latency_data:
        st.metric("P50", f"{latency_data['p50_ms']:.1f} ms")
        st.metric("P70", f"{latency_data['p70_ms']:.1f} ms")
        st.metric("P100", f"{latency_data['p100_ms']:.1f} ms")
        st.caption(f"Based on {latency_data['num_queries']} queries")
    else:
        st.caption("No latency data yet.")

    st.header("Settings")
    top_k = st.slider("Top-k results", 1, 10, 5)
    use_voice = st.checkbox("Enable voice input", value=False)

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
        if msg["role"] == "assistant" and "latency_ms" in msg:
            st.caption(f"Latency: {msg['latency_ms']:.0f}ms")

if use_voice:
    audio_file = st.audio_input("Record a question")
    if audio_file is not None:
        with st.chat_message("user"):
            st.audio(audio_file)
        with st.chat_message("assistant"):
            with st.spinner("Transcribing and thinking..."):
                try:
                    files = {"file": (audio_file.name, audio_file, audio_file.type)}
                    response = requests.post(
                        VOICE_URL,
                        files=files,
                        params={"top_k": top_k},
                        timeout=120,
                    )
                    response.raise_for_status()
                    data = response.json()

                    st.markdown(f"**Transcript:** {data['transcript']}")
                    st.markdown(data["answer"])

                    with st.expander("Sources"):
                        for s in data["sources"]:
                            st.markdown(
                                f"**chunk_id={s['chunk_id']}** "
                                f"(score={s['rerank_score']:.3f}) — {s['source_file']}"
                            )
                            st.caption(s["text_preview"])

                    st.caption(
                        f"STT: {data['stt_latency_ms']:.0f}ms | "
                        f"Total: {data['latency_ms']:.0f}ms | "
                        f"Guardrail: {'passed' if data['guardrail_passed'] else 'failed'}"
                    )

                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": f"🎙️ {data['transcript']}",
                        }
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": data["answer"],
                            "sources": data["sources"],
                            "latency_ms": data["latency_ms"],
                        }
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
else:
    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        QUERY_URL,
                        json={"question": prompt, "top_k": top_k},
                        timeout=120,
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
                        f"Latency: {data['latency_ms']:.0f}ms | "
                        f"Guardrail: {'passed' if data['guardrail_passed'] else 'failed'}"
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": data["answer"],
                            "sources": data["sources"],
                            "latency_ms": data["latency_ms"],
                        }
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
