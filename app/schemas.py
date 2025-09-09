from typing import List, Optional
from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    citations: Optional[List[str]] = None
    note: Optional[str] = None
    # NEW (observability)
    latency_ms: Optional[float] = None
    top_score: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
