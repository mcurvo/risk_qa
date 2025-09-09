#!/usr/bin/env python3
import os, json, pathlib
from typing import List, Dict
from dotenv import load_dotenv
from pypdf import PdfReader
import numpy as np
import faiss
from openai import OpenAI
import tiktoken

# 1) setup
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Missing OPENAI_API_KEY in .env")
client = OpenAI(api_key=api_key)

RAW_DIR = pathlib.Path("data/raw")
OUT_DIR = pathlib.Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
META_PATH = OUT_DIR / "meta.jsonl"
FAISS_PATH = OUT_DIR / "vectors.faiss"
EMBED_MODEL = "text-embedding-3-small"  # 1536-dim, cost-efficient

# 2) simple chunker: ~900 words, 200 words overlap
def chunk_text_tokenwise(text: str, max_tokens: int = 700, overlap_tokens: int = 150, model: str = "gpt-4o-mini") -> list:
    enc = tiktoken.encoding_for_model(model) if model in tiktoken.list_encoding_names() else tiktoken.get_encoding("cl100k_base")


    # Soft split by paragraphs first
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces = []
    for p in paras:
        toks = enc.encode(p)
        # If a single paragraph is huge, hard-split it
        if len(toks) <= max_tokens:
            pieces.append(toks)
        else:
            i = 0
            while i < len(toks):
                pieces.append(toks[i:i+max_tokens])
                i += max_tokens - overlap_tokens

    # Now merge small paragraphs until we’re close to max_tokens
    merged, cur = [], []
    for toks in pieces:
        if len(cur) + len(toks) <= max_tokens:
            cur += toks
        else:
            if cur:
                merged.append(cur)
                # start next chunk with overlap from end of previous
                cur = cur[-overlap_tokens:] + toks
            else:
                merged.append(toks)
                cur = []
    if cur:
        merged.append(cur)

    # Decode back to strings
    out = []
    for toks in merged:
        out.append(enc.decode(toks))
    return out

# 3) read PDF pages -> chunks with metadata
def pdf_to_chunks(pdf_path: pathlib.Path) -> List[Dict]:
    reader = PdfReader(str(pdf_path))
    out: List[Dict] = []
    for p, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        for ch in chunk_text_tokenwise(raw):
            ch = ch.strip()
            if ch:
                out.append({"text": ch, "source": pdf_path.name, "page": p})
    return out

# 4) embed in batches
def embed_texts(texts: List[str], batch: int = 128) -> np.ndarray:
    vecs: List[List[float]] = []
    for i in range(0, len(texts), batch):
        part = texts[i:i+batch]
        resp = client.embeddings.create(model=EMBED_MODEL, input=part)
        vecs.extend([d.embedding for d in resp.data])
    arr = np.array(vecs, dtype=np.float32)
    # normalize so dot product = cosine similarity
    faiss.normalize_L2(arr)
    return arr

def main():
    # gather chunks from all PDFs
    all_chunks: List[Dict] = []
    for p in sorted(RAW_DIR.glob("**/*.pdf")):
        print(f"Reading {p} ...")
        all_chunks.extend(pdf_to_chunks(p))

    if not all_chunks:
        raise SystemExit("No PDFs in data/raw. Add some and re-run.")

    texts = [c["text"] for c in all_chunks]
    print(f"Embedding {len(texts)} chunks ...")
    X = embed_texts(texts)

    # build FAISS index (inner product after normalization = cosine)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, str(FAISS_PATH))

    with META_PATH.open("w", encoding="utf-8") as f:
        for rec in all_chunks:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {FAISS_PATH} and {META_PATH}")

if __name__ == "__main__":
    main()
