from typing import List, Dict
from openai import OpenAI
import re, json
from .tools import lcr_ratio, toy_var

SYSTEM_PROMPT = (
    "You are a financial risk analyst (Basel; market, credit, liquidity, operational). "
    "Answer ONLY using the provided context snippets and/or approved tools. "
    "If the context is insufficient, say so. "
    "Be concise (4–7 sentences). Include inline citations like (source p.page) next to claims taken from the context."
)

_CITATION_RE = re.compile(r"\([^)]+ p\.\d+\)")

def build_context_block(snippets: List[Dict]) -> str:
    lines = [f"[{s['source']} p.{s['page']}] {s['text']}" for s in snippets]
    return "\n\n".join(lines)

def build_citations(snippets: List[Dict]) -> List[str]:
    seen, out = set(), []
    for s in snippets:
        tag = f"{s['source']} p.{s['page']}"
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out

def _has_inline_citation(text: str) -> bool:
    return bool(_CITATION_RE.search(text))

# OpenAI tool schema (function calling)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lcr_ratio",
            "description": "Compute Liquidity Coverage Ratio given HQLA and 30-day net cash outflows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hqla": {"type": "number", "description": "High-quality liquid assets amount"},
                    "net_outflows": {"type": "number", "description": "30-day total net cash outflows"},
                },
                "required": ["hqla", "net_outflows"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toy_var",
            "description": "Didactic Gaussian VaR calculator.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mean": {"type": "number"},
                    "stdev": {"type": "number"},
                    "horizon_days": {"type": "integer"},
                    "cl": {"type": "number", "description": "Confidence in (0,1), e.g., 0.99"},
                },
                "required": ["mean", "stdev", "horizon_days", "cl"],
            },
        },
    },
]

def _call_tool(name: str, args: Dict) -> Dict:
    if name == "lcr_ratio":
        return lcr_ratio(**args)
    if name == "toy_var":
        return toy_var(**args)
    return {"ok": False, "error": f"Unknown tool: {name}"}


def generate_answer(
    client: OpenAI,
    question: str,
    snippets: List[Dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
):
    """
    Returns: (content: str, usage: dict|None)
    usage has keys: prompt_tokens, completion_tokens, total_tokens (when provided by API)
    """
    context_block = build_context_block(snippets)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Context snippets (may be partial):\n{context_block}\n\n"
                "Use ONLY these snippets for regulatory definitions (with citations). "
                "If a small calculation is needed, you MAY call tools lcr_ratio or toy_var. "
                "If insufficient context for a definition, say so."
            ),
        },
    ]

    first = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=350,
    )
    msg = first.choices[0].message
    usage = getattr(first, "usage", None)
    usage_dict = None
    if usage:
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    if msg.tool_calls:
        tool_messages = []
        for tc in msg.tool_calls:
            import json as _json
            fn = tc.function.name
            args = _json.loads(tc.function.arguments or "{}")
            result = _call_tool(fn, args)
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fn,
                "content": _json.dumps(result),
            })

        follow = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages + [msg] + tool_messages,
            max_tokens=350,
        )
        content = follow.choices[0].message.content.strip()
        usage2 = getattr(follow, "usage", None)
        if usage2:
            usage_dict = {
                "prompt_tokens": getattr(usage2, "prompt_tokens", None),
                "completion_tokens": getattr(usage2, "completion_tokens", None),
                "total_tokens": getattr(usage2, "total_tokens", None),
            }
    else:
        content = msg.content.strip() if msg.content else ""

    # Post-condition: require at least one inline citation for definition-like queries
    needs_cite = any(w in question.lower() for w in ["what", "define", "definition", "basel", "lcr", "ratio", "coverage"])
    if needs_cite and not _has_inline_citation(content):
        return ("Insufficient grounded context to answer confidently from the provided documents. "
                "Please provide additional materials or specify the exact section/page to consult."), usage_dict

    return content, usage_dict
