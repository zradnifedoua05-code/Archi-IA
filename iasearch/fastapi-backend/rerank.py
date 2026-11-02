"""
TD3 — Prompt du cours :
adds rerank function which uses ollama locally on http://localhost:11434/ with model
qwen3.5:0.8b, constrain output to JSON yes/no enum, find the answer token's logprobs,
sum yes/no variant probabilities, score p_yes/(p_yes+p_no), sort descending
"""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx

RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3.5:0.8b")
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Schema exact du slide partie 4 :
# {"type":"object","properties":{"answer":{"type":"string","enum":["yes","no"]}},"required":["answer"]}
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["yes", "no"]},
    },
    "required": ["answer"],
}


def _normalize_token(token: str) -> str:
    return token.strip().lower().strip('"').strip("'")


def _score_from_logprobs(logprobs: list[dict[str, Any]] | None) -> float | None:
    """Sum yes/no variant probabilities → p_yes / (p_yes + p_no)."""
    if not logprobs:
        return None

    p_yes = 0.0
    p_no = 0.0

    for entry in logprobs:
        candidates: list[dict[str, Any]] = []
        if entry.get("top_logprobs"):
            candidates.extend(entry["top_logprobs"])
        if "token" in entry and "logprob" in entry:
            candidates.append(
                {"token": entry["token"], "logprob": entry["logprob"]}
            )

        for cand in candidates:
            token = _normalize_token(str(cand.get("token", "")))
            logprob = float(cand.get("logprob", float("-inf")))
            if not math.isfinite(logprob):
                continue
            prob = math.exp(logprob)
            if token in {"yes", "y", "oui"}:
                p_yes += prob
            elif token in {"no", "n", "non"}:
                p_no += prob

    # Need both sides for a meaningful soft score. With the object JSON schema,
    # Ollama often only returns logprobs for the first "{" token; yes/no may be
    # missing → fall back to the constrained answer instead of a biased 1.0.
    if p_yes <= 0 or p_no <= 0:
        return None
    return p_yes / (p_yes + p_no)


def _parse_answer(raw: Any) -> str:
    if isinstance(raw, dict):
        return str(raw.get("answer", "no")).lower()
    text = str(raw or "").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return str(parsed.get("answer", "no")).lower()
        if isinstance(parsed, str):
            return parsed.lower().strip()
    except json.JSONDecodeError:
        pass
    lowered = text.lower()
    if "yes" in lowered:
        return "yes"
    if "no" in lowered:
        return "no"
    return "no"


def _score_document(
    query: str, document: str, ollama_host: str
) -> dict[str, Any]:
    prompt = (
        "You are a relevance judge. Decide if the document is relevant to the query.\n"
        f"Query: {query}\n"
        f"Document: {document}\n"
        'Answer with JSON only: {"answer":"yes"} or {"answer":"no"}.'
    )

    # Constrained JSON yes/no enum + logprobs (slide partie 4).
    # think=False: requis pour qwen3.5 sinon response vide (mode reasoning).
    # logprobs top-level: requis Ollama >= 0.40 (pas dans options).
    payload = {
        "model": RERANK_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": JSON_SCHEMA,
        "think": False,
        "logprobs": True,
        "top_logprobs": 10,
        "options": {
            "temperature": 0,
            "num_predict": 32,
        },
    }

    response = httpx.post(
        f"{ollama_host}/api/generate",
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()

    raw = data.get("response", "")
    answer = _parse_answer(raw)
    score = _score_from_logprobs(data.get("logprobs"))
    if score is None:
        score = 1.0 if answer == "yes" else 0.0

    return {
        "document": document,
        "answer": answer,
        "score": score,
        "raw": raw,
    }


def rerank(
    query: str,
    documents: list[str],
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """Score chaque document en parallèle, trie par score décroissant."""
    if not documents:
        return []

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_score_document, query, doc, ollama_host): doc
            for doc in documents
        }
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item["score"], reverse=True)
    return results
