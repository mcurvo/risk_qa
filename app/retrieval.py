import json, pathlib, os
from typing import List, Dict
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = pathlib.Path("data/processed")
META_PATH = DATA_DIR / "meta.jsonl"
FAISS_PATH = DATA_DIR / "vectors.faiss"
EMBED_MODEL = "text-embedding-3-small"

_index = None
_meta: List[Dict] = []
_client = None

def _client_instance():
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY missing")
        _client = OpenAI(api_key=key)
    return _client

def load_meta() -> List[Dict]:
    global _meta
    if _meta:
        return _meta
    items: List[Dict] = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    _meta = items
    return _meta

def load_index():
    global _index
    if _index is None:
        _index = faiss.read_index(str(FAISS_PATH))
    return _index

def embed_query(q: str) -> np.ndarray:
    client = _client_instance()
    v = client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding
    arr = np.array([v], dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr

def retrieve(q: str, k: int = 5) -> List[Dict]:
    meta = load_meta()
    index = load_index()
    v = embed_query(q)                  # (1, d)
    # wider initial search
    k_cand = max(5*k, 20)
    scores, idxs = index.search(v, k_cand)

    # Filter valid indices and collect candidate embeddings (we’ll need vectors)
    cand = []
    vecs = []
    for score, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        rec = meta[int(i)].copy()
        rec["score"] = float(score)
        cand.append((int(i), rec))
    if not cand:
        return []

    # We need the actual vectors for MMR; IndexFlatIP doesn’t expose them directly,
    # so we re-embed the candidate texts (small cost for better ranking).
    texts = [rec["text"] for _, rec in cand]
    client = _client_instance()
    embs = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in embs.data], dtype=np.float32)
    faiss.normalize_L2(vecs)

    # Apply MMR to choose k diverse, relevant chunks
    sel = mmr_select(vecs, v, k=k, lambda_param=0.7)

    out = []
    for j in sel:
        out.append(cand[j][1])  # rec with score/source/page/text
    return out

def mmr_select(embeddings: np.ndarray, query_vec: np.ndarray, k: int, lambda_param: float = 0.7):
    """
    embeddings: (N, d) normalized chunk vectors
    query_vec:  (1, d) normalized query vector
    Returns indices of k items using Maximal Marginal Relevance.
    """
    # Cosine sims since everything is L2-normalized
    sims_to_query = (embeddings @ query_vec.T).ravel()
    selected = []
    candidates = set(range(embeddings.shape[0]))

    # seed with the best
    first = int(np.argmax(sims_to_query))
    selected.append(first)
    candidates.remove(first)

    while len(selected) < min(k, embeddings.shape[0]):
        best_idx, best_score = None, -1e9
        for i in candidates:
            relevance = sims_to_query[i]
            diversity = max((embeddings[i] @ embeddings[j] for j in selected), default=0.0)
            score = lambda_param * relevance - (1 - lambda_param) * diversity
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected
