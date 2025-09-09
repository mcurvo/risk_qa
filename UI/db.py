import sqlite3, json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

DEFAULT_DB_PATH = "data/chat_logs.db"

DDL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_key TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0,   -- NEW
    share_token TEXT                       -- NEW (nullable)
);
CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations(owner_key);
CREATE INDEX IF NOT EXISTS idx_conversations_arch ON conversations(archived);

CREATE TABLE IF NOT EXISTS conv_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations TEXT,
    latency_ms REAL,
    top_score REAL,
    raw_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_msgs_convo ON conv_messages(conversation_id);
"""

def now() -> str:
    return datetime.utcnow().isoformat() + "Z"

def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.executescript(DDL)
    return conn

def create_conversation(conn: sqlite3.Connection, owner_key: str, title: str = "New chat") -> int:
    cur = conn.cursor()
    ts = now()
    cur.execute(
        "INSERT INTO conversations (owner_key, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (owner_key, title, ts, ts),
    )
    conn.commit()
    return int(cur.lastrowid)

def rename_conversation(conn: sqlite3.Connection, convo_id: int, new_title: str):
    cur = conn.cursor()
    cur.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?", (new_title, now(), convo_id))
    conn.commit()

def delete_conversation(conn: sqlite3.Connection, convo_id: int):
    cur = conn.cursor()
    cur.execute("DELETE FROM conv_messages WHERE conversation_id=?", (convo_id,))
    cur.execute("DELETE FROM conversations WHERE id=?", (convo_id,))
    conn.commit()

def get_messages(conn: sqlite3.Connection, convo_id: int) -> List[Dict]:
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content, citations, latency_ms, top_score, raw_json, created_at "
        "FROM conv_messages WHERE conversation_id=? ORDER BY id ASC",
        (convo_id,),
    )
    rows = cur.fetchall()
    msgs: List[Dict] = []
    for role, content, citations, latency_ms, top_score, raw_json, created_at in rows:
        item = {"role": role, "content": content, "created_at": created_at}
        if citations:
            try: item["citations"] = json.loads(citations)
            except Exception: item["citations"] = []
        if latency_ms is not None: item["latency_ms"] = latency_ms
        if top_score is not None: item["top_score"] = top_score
        if raw_json: item["raw_json"] = raw_json
        msgs.append(item)
    return msgs

def append_user_message(conn: sqlite3.Connection, convo_id: int, content: str):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO conv_messages (conversation_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
        (convo_id, content, now()),
    )
    cur.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now(), convo_id))
    conn.commit()

def append_assistant_message(conn: sqlite3.Connection, convo_id: int, answer_json: dict):
    cur = conn.cursor()
    citations = json.dumps(answer_json.get("citations", []))
    cur.execute(
        "INSERT INTO conv_messages (conversation_id, role, content, citations, latency_ms, top_score, raw_json, created_at) "
        "VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?)",
        (
            convo_id,
            answer_json.get("answer", ""),
            citations,
            float(answer_json.get("latency_ms")) if answer_json.get("latency_ms") is not None else None,
            float(answer_json.get("top_score")) if answer_json.get("top_score") is not None else None,
            json.dumps(answer_json),
            now(),
        ),
    )
    cur.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now(), convo_id))
    conn.commit()

def first_user_to_title(prompt: str, max_len: int = 60) -> str:
    t = prompt.strip().splitlines()[0]
    return (t[:max_len] + "…") if len(t) > max_len else t


def list_conversations(conn, owner_key: str, limit: int = 50, include_archived: bool = False):
    cur = conn.cursor()
    if include_archived:
        cur.execute(
            "SELECT id, title, updated_at, archived FROM conversations "
            "WHERE owner_key=? ORDER BY archived ASC, updated_at DESC LIMIT ?",
            (owner_key, limit),
        )
    else:
        cur.execute(
            "SELECT id, title, updated_at, archived FROM conversations "
            "WHERE owner_key=? AND archived=0 ORDER BY updated_at DESC LIMIT ?",
            (owner_key, limit),
        )
    return cur.fetchall()

def archive_conversation(conn, convo_id: int, archived: bool = True):
    cur = conn.cursor()
    cur.execute("UPDATE conversations SET archived=?, updated_at=? WHERE id=?",
                (1 if archived else 0, now(), convo_id))
    conn.commit()

def gen_share_token():
    import secrets
    return secrets.token_urlsafe(12)

def ensure_share_token(conn, convo_id: int) -> str:
    cur = conn.cursor()
    cur.execute("SELECT share_token FROM conversations WHERE id=?", (convo_id,))
    row = cur.fetchone()
    if row and row[0]:
        return row[0]
    token = gen_share_token()
    cur.execute("UPDATE conversations SET share_token=?, updated_at=? WHERE id=?",
                (token, now(), convo_id))
    conn.commit()
    return token
