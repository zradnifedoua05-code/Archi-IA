# IASearch — Projet d’évaluation Architectures IA

Rendu : archive ou lien GitHub/GitLab → `nicolas.debeissat@mail-formateur.net`

Conformité détaillée (chaque prompt du PDF) : [`CONFORMITE_TP.md`](./CONFORMITE_TP.md)

## Architecture (partie 1)

```
Front React → FastAPI → Elasticsearch (clusters + docs)
                     ↘ Ollama embedding / rerank
Script index_data → embeddings Ollama → KMeans → index ES
```

## Prérequis

Docker Desktop, [uv](https://docs.astral.sh/uv/), Node.js (dev front).

## Démarrage (sans échec)

```bash
cd iasearch

# Pré-télécharger les images (évite les timeouts Docker Hub)
docker pull docker.elastic.co/elasticsearch/elasticsearch:9.3.2
docker pull ollama/ollama:latest
docker pull node:22-alpine
docker pull nginx:1.27-alpine
docker pull ghcr.io/astral-sh/uv:python3.12-trixie-slim

docker compose up -d --build
```

| Service | URL |
|---------|-----|
| Elasticsearch | http://localhost:9200 |
| Ollama | http://localhost:11434 |
| API Swagger | http://localhost:8000/docs |
| Front | http://localhost:3000 |

### Modèles Ollama (TD1-2 + TD3)

```bash
docker exec -it ollama ollama pull qwen3-embedding:0.6b
docker exec -it ollama ollama pull qwen3.5:0.8b
```

### GPU Linux (prompt TD1-2)

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### Indexation (TD1-3)

```bash
cd index_data
uv sync
uv run python main.py
```

### Notebook Colab (TD1-1)

Ouvrir `td1_colab/TD1_1_embeddings_clustering.ipynb` dans Google Colab.

## Debug (TD2-2 / TD2-3)

Configs VS Code : `.vscode/launch.json`

- **Python Debugger: FastAPI** — local
- **Python Debugger: Remote Attach** — conteneur port `5678`

## Parcours des TDs

1. **TD1-1** notebook embeddings HF + projector + KMeans  
2. **TD1-2** ES + Ollama  
3. **TD1-3** `index_data` → `data_*` + `data_clusters`  
4. **TD2** FastAPI KNN + hybride + front Search  
5. **TD3** `rerank.py` logprobs yes/no  
