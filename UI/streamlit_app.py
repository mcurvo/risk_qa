# ui/streamlit_app.py
import os
import requests
import streamlit as st

API_BASE = os.getenv("RISKQA_API_BASE", "http://localhost:8000")

def get_ip() -> str:
    try:
        r = requests.get(f"{API_BASE}/whoami", timeout=3)
        if r.ok and "ip" in r.json():
            return r.json()["ip"]
    except Exception:
        pass
    return "unknown"

def ask_api(question: str) -> dict:
    r = requests.post(f"{API_BASE}/ask", json={"question": question}, timeout=60)
    r.raise_for_status()
    return r.json()

# ---------- UI ----------
st.set_page_config(page_title="Risk Q&A", page_icon="💬", layout="centered")
st.title("💬 LLM-Powered Financial Risk Q&A")

ip = get_ip()
st.caption(f"Detected IP: `{ip}`  •  API base: {API_BASE}")

# Keep chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": str}]

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# Chat input
prompt = st.chat_input("Ask a question about financial risk…")
if prompt:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Call backend and display assistant message
    try:
        resp = ask_api(prompt)
        answer = resp.get("answer", "")
        cites = resp.get("citations") or []
        latency = resp.get("latency_ms")
        top_score = resp.get("top_score")

        with st.chat_message("assistant"):
            st.write(answer)
            if cites:
                st.caption("Sources: " + " • ".join(cites))
            extra = []
            if latency is not None: extra.append(f"Latency: {latency:.0f} ms")
            if top_score is not None: extra.append(f"Score: {top_score:.2f}")
            if extra: st.caption("  |  ".join(extra))

        st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error contacting API: {e}")

st.sidebar.header("Help")
st.sidebar.write("Set `RISKQA_API_BASE` to point at your FastAPI server.")
