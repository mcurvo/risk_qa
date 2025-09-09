from fastapi import FastAPI, Body, HTTPException, Request
from dotenv import load_dotenv
import os, time, pathlib

from .schemas import AskRequest, AskResponse
from .retrieval import retrieve
from .generate import generate_answer, build_citations
from openai import OpenAI

load_dotenv()
app = FastAPI(title="Financial Risk Q&A")

INDEX_FILES = [pathlib.Path("data/processed/vectors.faiss"), pathlib.Path("data/processed/meta.jsonl")]

@app.get("/health")
def health():
    return {"ok": True, "stage": "step-6"}

@app.get("/health/ready")
def health_ready():
    """Stronger readiness check: index files + optional API key."""
    index_ok = all(p.exists() for p in INDEX_FILES)
    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    return {
        "ok": index_ok,
        "index_ok": index_ok,
        "api_key_present": api_key_present,
        "details": "index_ok requires data/processed/{vectors.faiss, meta.jsonl}",
    }

def _client_or_none():
    key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=key) if key else None

def _dedup_and_trim(cites, keep: int = 5):
    seen, out = set(), []
    for c in cites:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:keep]

@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest = Body(...)):
    t0 = time.time()

    # 1) Retrieve top-k
    try:
        ctx = retrieve(payload.question, k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    if not ctx:
        latency = (time.time() - t0) * 1000
        return AskResponse(
            answer="I couldn’t find relevant passages in your index.",
            note="Add PDFs to data/raw and rebuild the index.",
            latency_ms=latency
        )

    # 2) Confidence gate
    top_score = float(ctx[0].get("score", 0.0))
    LOW_CONF = 0.35
    if top_score < LOW_CONF:
        latency = (time.time() - t0) * 1000
        return AskResponse(
            answer=(
                "The retrieved context is weak or off-topic, so I won't speculate. "
                "Please provide more relevant documents or rephrase the question."
            ),
            citations=_dedup_and_trim(build_citations(ctx)),
            note=f"Top similarity {top_score:.2f} < {LOW_CONF:.2f}.",
            latency_ms=latency,
            top_score=top_score
        )

    client = _client_or_none()
    if client is None:
        latency = (time.time() - t0) * 1000
        best = ctx[0]["text"]
        return AskResponse(
            answer=f"(DEV MODE: no OPENAI_API_KEY) Closest context says: {best[:700]}{'…' if len(best)>700 else ''}",
            citations=_dedup_and_trim(build_citations(ctx)),
            note="Set OPENAI_API_KEY in .env to enable grounded LLM answers.",
            latency_ms=latency,
            top_score=top_score
        )

    try:
        content, usage = generate_answer(client, payload.question, ctx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    latency = (time.time() - t0) * 1000
    return AskResponse(
        answer=content,
        citations=_dedup_and_trim(build_citations(ctx)),
        latency_ms=latency,
        top_score=top_score,
        prompt_tokens=(usage or {}).get("prompt_tokens"),
        completion_tokens=(usage or {}).get("completion_tokens"),
        total_tokens=(usage or {}).get("total_tokens"),
    )

@app.get("/whoami")
def whoami(request: Request):
    # If behind a proxy / load balancer, prefer X-Forwarded-For (first in list)
    ip = request.headers.get("x-forwarded-for")
    if ip:
        ip = ip.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"
    return {"ip": ip}
