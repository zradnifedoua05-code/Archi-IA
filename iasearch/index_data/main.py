"""
TD1-3 — Indexer les données (reprendre le notebook Collab dans main.py)

lines = get_data_from_cloud()
line_embeddings = get_data_embeddings(lines)
cluster_to_lines = cluster_data(line_embeddings, lines)

Puis indexer dans Elasticsearch :
- 1 index par cluster data_1, data_2, ... : { embedding, line }
- index data_clusters : { embedding, data_index }
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

import numpy as np
import requests
from elasticsearch import Elasticsearch, helpers
from sklearn.cluster import KMeans

# Même URL que TD1-1 cellule 1 (pCloud)
PCLOUD_META_URL = (
    "https://apilux3.pcloud.com/getpublinkdownload"
    "?fileid=93128819642&code=XZnqkRJZ98VUd3rEuMXAFIsEpAtnNHoKYK8k"
    "&linkpassword=&forcedownload=1"
)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
N_CLUSTERS = int(os.getenv("N_CLUSTERS", "10"))
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))


def get_data_from_cloud() -> list[str]:
    """Download consultation titles from the public pCloud link (TD1-1)."""
    meta = requests.get(PCLOUD_META_URL, timeout=60).json()
    if meta.get("result") != 0:
        raise RuntimeError(f"pCloud metadata error: {meta}")

    path = meta["path"]
    host = meta["hosts"][0]
    download_url = f"https://{host}{path}"
    print(f"Downloading dataset from {download_url}")

    response = requests.get(download_url, timeout=120)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"

    lines = [line.strip() for line in response.text.splitlines() if line.strip()]
    print(f"Loaded {len(lines)} lines")
    return lines


def get_data_embeddings(lines: list[str]) -> np.ndarray:
    """
    TD1-3 : calculer les embeddings via docker Ollama
    http://localhost:11434  (modèle qwen3-embedding:0.6b)
    """
    embeddings: list[list[float]] = []
    total = len(lines)

    for start in range(0, total, BATCH_SIZE):
        batch = lines[start : start + BATCH_SIZE]
        response = requests.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": batch},
            timeout=300,
        )
        response.raise_for_status()
        batch_embeddings = response.json()["embeddings"]
        if len(batch_embeddings) != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} embeddings, got {len(batch_embeddings)}"
            )
        embeddings.extend(batch_embeddings)
        done = min(start + BATCH_SIZE, total)
        print(f"Embeddings: {done}/{total}")

    return np.array(embeddings, dtype=np.float32)


def cluster_data(
    line_embeddings: np.ndarray, lines: list[str], n_clusters: int = N_CLUSTERS
) -> dict[str, Any]:
    """KMeans (TD1-1 cellule 4) — retourne clusters + centroïdes."""
    print(f"Running KMeans with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(line_embeddings)

    cluster_to_lines: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for label, line, emb in zip(labels, lines, line_embeddings):
        cluster_to_lines[int(label)].append(
            {"line": line, "embedding": emb.tolist()}
        )

    centroids = {
        int(i): kmeans.cluster_centers_[i].tolist() for i in range(n_clusters)
    }
    return {"clusters": dict(cluster_to_lines), "centroids": centroids}


def wait_for_elasticsearch(es: Elasticsearch, retries: int = 60) -> None:
    for attempt in range(retries):
        try:
            if es.ping():
                print("Elasticsearch is ready")
                return
        except Exception:
            pass
        print(f"Waiting for Elasticsearch... ({attempt + 1}/{retries})")
        time.sleep(2)
    raise RuntimeError("Elasticsearch did not become ready in time")


def index_to_elasticsearch(
    cluster_payload: dict[str, Any], es: Elasticsearch
) -> None:
    """
    Indexer les lignes des clusters : data_1, data_2, ...
      { "embedding": [...], "line": "RD2E7; Travaux de..." }
    Indexer data_clusters :
      { "embedding": [...], "data_index": "data_1" }
    """
    clusters = cluster_payload["clusters"]
    centroids = cluster_payload["centroids"]
    dims = len(next(iter(centroids.values())))

    try:
        cat = es.cat.indices(index="data_*", format="json")
        existing = [row["index"] for row in cat]
    except Exception:
        existing = []

    for index_name in existing:
        print(f"Deleting index {index_name}")
        es.indices.delete(index=index_name, ignore_unavailable=True)

    dense_mapping = {
        "properties": {
            "embedding": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine",
            },
            "line": {"type": "text"},
            "data_index": {"type": "keyword"},
        }
    }

    for cluster_id in sorted(clusters.keys()):
        index_name = f"data_{cluster_id + 1}"
        print(f"Creating index {index_name} ({len(clusters[cluster_id])} docs)")
        es.indices.create(index=index_name, mappings=dense_mapping)

        actions = [
            {
                "_index": index_name,
                "_source": {
                    "embedding": item["embedding"],
                    "line": item["line"],
                },
            }
            for item in clusters[cluster_id]
        ]
        helpers.bulk(es, actions)
        es.indices.refresh(index=index_name)

    print("Creating index data_clusters")
    es.indices.create(index="data_clusters", mappings=dense_mapping)

    cluster_actions = [
        {
            "_index": "data_clusters",
            "_source": {
                "embedding": centroids[cluster_id],
                "data_index": f"data_{cluster_id + 1}",
            },
        }
        for cluster_id in sorted(centroids.keys())
    ]
    helpers.bulk(es, cluster_actions)
    es.indices.refresh(index="data_clusters")
    print("Indexing complete")


def main() -> None:
    lines = get_data_from_cloud()
    line_embeddings = get_data_embeddings(lines)
    cluster_to_lines = cluster_data(line_embeddings, lines)

    es = Elasticsearch(ELASTICSEARCH_HOST)
    wait_for_elasticsearch(es)
    index_to_elasticsearch(cluster_to_lines, es)

    print(
        f"Done. {len(lines)} documents in {len(cluster_to_lines['clusters'])} clusters."
    )


if __name__ == "__main__":
    main()
