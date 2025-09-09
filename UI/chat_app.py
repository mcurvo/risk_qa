# ui/chat_app.py
import os, time, requests, streamlit as st
from UI.db import connect, list_conversations, create_conversation, rename_conversation, delete_conversation, \
                  get_messages, append_user_message, append_assistant_message, first_user_to_title, DEFAULT_DB_PATH

API_BASE = os.getenv("RISKQA_API_BASE", "http://localhost:8000")
DB_PATH = os.getenv("RISKQA_DB_PATH", DEFAULT_DB_PATH)

st.set_page_config(page_title="Risk Q&A", page_icon="💬", layout="wide")

st.session_state.setdefault("auth_user", None)
st.session_state.setdefault("current_convo_id", None)
st.session_state.setdefault("new_chat_mode", False)
st.session_state.setdefault("menu_open_for", None)

def get_ip() -> str:
    try:
        r = requests.get(f"{API_BASE}/whoami", timeout=3)
        if r.ok and "ip" in r.json(): return r.json()["ip"]
    except Exception:
        pass
    return "unknown"

def ask_api(question: str) -> dict:
    r = requests.post(f"{API_BASE}/ask", json={"question": question}, timeout=120)
    r.raise_for_status()
    return r.json()

# --- session init ---
if "conn" not in st.session_state:
    st.session_state.conn = connect(DB_PATH)
if "client_ip" not in st.session_state:
    st.session_state.client_ip = get_ip()
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

def owner_key() -> str:
    return f"user:{st.session_state.auth_user}" if st.session_state.auth_user else f"ip:{st.session_state.client_ip}"

st.sidebar.markdown("### 💬 Chats")
conn = st.session_state.conn
okey = owner_key()

# Controls
c1, c2, c3 = st.sidebar.columns([1, 1, 1])
if c1.button("➕ New"):
    st.session_state.current_convo_id = None
    st.session_state.new_chat_mode = True
if c2.button("🗑️ Delete", disabled=st.session_state.current_convo_id is None):
    if st.session_state.current_convo_id is not None:
        delete_conversation(conn, st.session_state.current_convo_id)
        st.session_state.current_convo_id = None
        st.session_state.new_chat_mode = True
if c3.button("🔄 Refresh"):
    pass

show_archived = st.sidebar.toggle("Show archived", value=False, help="Include archived chats in the list")

# Track which menu is open (by conversation id)
if "menu_open_for" not in st.session_state:
    st.session_state.menu_open_for = None

convos = list_conversations(conn, okey, limit=100, include_archived=show_archived)

def select_chat(cid: int):
    st.session_state.current_convo_id = cid
    st.session_state.new_chat_mode = False

# Render list (title + three-dots)
for cid, title, updated_at, archived in convos:
    row = st.sidebar.container()
    cols = row.columns([0.85, 0.15])
    # title button selects
    if cols[0].button(title, key=f"sel_{cid}", use_container_width=True):
        select_chat(cid)
        st.session_state.menu_open_for = None

    # three-dots opens menu
    if cols[1].button("⋯", key=f"menu_{cid}"):
        st.session_state.menu_open_for = cid if st.session_state.menu_open_for != cid else None

    # actions menu (inline)
    if st.session_state.menu_open_for == cid:
        box = st.sidebar.container(border=True)
        b1, b2 = box.columns(2)
        if b1.button("Rename", key=f"rename_{cid}"):
            st.session_state[f"rename_mode_{cid}"] = True
        if b2.button("Archive" if not archived else "Unarchive", key=f"arch_{cid}"):
            archive_conversation(conn, cid, archived=(not archived))
            st.session_state.menu_open_for = None

        b3, b4 = box.columns(2)
        if b3.button("Delete", key=f"delete_{cid}"):
            delete_conversation(conn, cid)
            if st.session_state.current_convo_id == cid:
                st.session_state.current_convo_id = None
            st.session_state.menu_open_for = None
        if b4.button("Share", key=f"share_{cid}"):
            token = ensure_share_token(conn, cid)
            share_url = f"{API_BASE}/share/{token}"  # Stub: display URL; implement endpoint later if you like
            st.sidebar.info(f"Share URL:\n{share_url}")
            st.session_state.menu_open_for = None

        # inline rename UI
        if st.session_state.get(f"rename_mode_{cid}"):
            new_t = box.text_input("New title", value=title, key=f"rename_input_{cid}")
            if box.button("Save title", key=f"rename_save_{cid}"):
                nt = new_t.strip()
                if nt and nt != title:
                    rename_conversation(conn, cid, nt)
                st.session_state[f"rename_mode_{cid}"] = False

st.title("💬 LLM-Powered Financial Risk Q&A")
st.caption("Ask about Basel/IFRS and your uploaded risk docs. Answers are grounded with citations.")

cid = st.session_state.current_convo_id

if cid is None:
    st.info("Start a new chat by asking your first question. I’ll create and save it when you send.")
else:
    msgs = get_messages(st.session_state.conn, cid)
    for m in msgs:
        role = m["role"] if m["role"] in ("user", "assistant") else "assistant"
        with st.chat_message(role):
            st.write(m["content"])
            if m.get("citations"):
                st.caption("Sources: " + " • ".join(m["citations"]))
            extras = []
            if m.get("latency_ms") is not None: extras.append(f"Latency: {m['latency_ms']:.0f} ms")
            if m.get("top_score") is not None: extras.append(f"Score: {m['top_score']:.2f}")
            if extras: st.caption("  |  ".join(extras))

# Chat input
prompt = st.chat_input("Ask a question about financial risk…")

if prompt:
    # If we don't have a conversation yet, create it now, titled from the first message
    if st.session_state.current_convo_id is None:
        title = first_user_to_title(prompt)
        st.session_state.current_convo_id = create_conversation(st.session_state.conn, owner_key(), title)
        st.session_state.new_chat_mode = False

    # Now persist + render user message
    append_user_message(st.session_state.conn, st.session_state.current_convo_id, prompt)
    with st.chat_message("user"):
        st.write(prompt)

    # call backend
    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=False)
        try:
            t0 = time.time()
            resp = ask_api(prompt)
            dt = (time.time() - t0) * 1000.0
            latency = resp.get("latency_ms", dt)

            st.write(resp.get("answer",""))
            if resp.get("citations"):
                st.caption("Sources: " + " • ".join(resp["citations"]))
            extra = []
            if latency is not None: extra.append(f"Latency: {latency:.0f} ms")
            if resp.get("top_score") is not None: extra.append(f"Score: {resp['top_score']:.2f}")
            if extra: st.caption("  |  ".join(extra))
            status.update(label="Done", state="complete", expanded=False)

            # persist assistant message
            append_assistant_message(conn, st.session_state.current_convo_id, resp)

            # auto-title on first question
            if len(get_messages(conn, st.session_state.current_convo_id)) <= 2:
                rename_conversation(conn, st.session_state.current_convo_id, first_user_to_title(prompt))
        except Exception as e:
            st.error(f"Error contacting API: {e}")
            status.update(label="Error", state="error", expanded=True)
