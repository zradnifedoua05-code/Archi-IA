"""TD2 / TD3: FastAPI search backend with KNN, hybrid search and LLM rerank."""

from __future__ import annotations

import os
from typing import Any

import httpx
from elasticsearch import Elasticsearch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rerank import rerank

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")

app = FastAPI(title="IASearch", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

es = Elasticsearch(ELASTICSEARCH_HOST)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_clusters: int = Field(default=3, ge=1, le=20)
    top_docs: int = Field(default=10, ge=1, le=50)
    use_rerank: bool = True


class SearchHit(BaseModel):
    line: str
    score: float
    cluster_index: str
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    clusters: list[str]
    results: list[SearchHit]


def get_query_embedding(text: str) -> list[float]:
    response = httpx.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=120.0,
    )
    response.raise_for_status()
    embeddings = response.json()["embeddings"]
    return embeddings[0]


def search_clusters(embedding: list[float], k: int) -> list[dict[str, Any]]:
    """KNN search over data_clusters to find nearest cluster indices."""
    body = {
        "knn": {
            "field": "embedding",
            "query_vector": embedding,
            "k": k,
            "num_candidates": max(k * 10, 50),
        },
        "_source": ["data_index"],
    }
    result = es.search(index="data_clusters", body=body)
    hits = []
    for hit in result["hits"]["hits"]:
        hits.append(
            {
                "data_index": hit["_source"]["data_index"],
                "score": hit["_score"],
            }
        )
    return hits


def hybrid_search_in_cluster(
    index_name: str, query: str, embedding: list[float], size: int
) -> list[dict[str, Any]]:
    """Hybrid BM25 + KNN search inside a cluster index."""
    body = {
        "size": size,
        "query": {
            "bool": {
                "should": [
                    {"match": {"line": {"query": query, "boost": 1.0}}},
                ]
            }
        },
        "knn": {
            "field": "embedding",
            "query_vector": embedding,
            "k": size,
            "num_candidates": max(size * 10, 50),
            "boost": 1.0,
        },
        "_source": ["line"],
    }
    result = es.search(index=index_name, body=body)
    return [
        {
            "line": hit["_source"]["line"],
            "score": hit["_score"],
            "cluster_index": index_name,
        }
        for hit in result["hits"]["hits"]
    ]


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "iasearch-backend"}


@app.get("/health")
def health() -> dict[str, Any]:
    es_ok = False
    try:
        es_ok = bool(es.ping())
    except Exception:
        es_ok = False
    return {
        "elasticsearch": es_ok,
        "elasticsearch_host": ELASTICSEARCH_HOST,
        "ollama_host": OLLAMA_HOST,
    }


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    try:
        embedding = get_query_embedding(query)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Ollama embedding failed: {exc}"
        ) from exc

    try:
        cluster_hits = search_clusters(embedding, payload.top_clusters)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Cluster search failed: {exc}"
        ) from exc

    if not cluster_hits:
        return SearchResponse(query=query, clusters=[], results=[])

    cluster_indices = [c["data_index"] for c in cluster_hits]
    docs: list[dict[str, Any]] = []
    for index_name in cluster_indices:
        try:
            docs.extend(
                hybrid_search_in_cluster(
                    index_name, query, embedding, payload.top_docs
                )
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Hybrid search failed on {index_name}: {exc}",
            ) from exc

    # Deduplicate by line text, keep highest ES score
    best: dict[str, dict[str, Any]] = {}
    for doc in docs:
        key = doc["line"]
        if key not in best or doc["score"] > best[key]["score"]:
            best[key] = doc
    unique_docs = sorted(best.values(), key=lambda d: d["score"], reverse=True)

    results: list[SearchHit]
    if payload.use_rerank and unique_docs:
        documents = [d["line"] for d in unique_docs]
        try:
            ranked = rerank(query, documents, ollama_host=OLLAMA_HOST)
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"Rerank failed: {exc}"
            ) from exc

        score_by_line = {item["document"]: item["score"] for item in ranked}
        results = [
            SearchHit(
                line=d["line"],
                score=d["score"],
                cluster_index=d["cluster_index"],
                rerank_score=score_by_line.get(d["line"]),
            )
            for d in unique_docs
        ]
        results.sort(
            key=lambda h: (h.rerank_score if h.rerank_score is not None else 0.0),
            reverse=True,
        )
        results = results[: payload.top_docs]
    else:
        results = [
            SearchHit(
                line=d["line"],
                score=d["score"],
                cluster_index=d["cluster_index"],
            )
            for d in unique_docs[: payload.top_docs]
        ]

    return SearchResponse(
        query=query, clusters=cluster_indices, results=results
    )


@app.get("/search")
def search_get(
    q: str = Query(..., min_length=1),
    top_clusters: int = Query(3, ge=1, le=20),
    top_docs: int = Query(10, ge=1, le=50),
    use_rerank: bool = Query(True),
) -> SearchResponse:
    return search(
        SearchRequest(
            query=q,
            top_clusters=top_clusters,
            top_docs=top_docs,
            use_rerank=use_rerank,
        )
    )
