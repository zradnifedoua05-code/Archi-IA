# Conformité TP Architectures IA (4 parties)

Mail de rendu : `nicolas.debeissat@mail-formateur.net`  
Livrable : archive ou repo GitHub/GitLab du projet `iasearch` dockerisé.

## Pourquoi Docker a échoué

Le build du service `front` a renvoyé :

`DeadlineExceeded` en tirant `node:22-alpine` et `nginx:1.27-alpine` depuis `registry-1.docker.io`.

Cause : timeout réseau vers Docker Hub (pas une erreur de Dockerfile).  
Correction : pré-télécharger les images (`docker pull`), puis `docker compose up -d --build`.

---

## Partie 1 — Cours + projet d’évaluation

| Exigence | Couverture |
|----------|------------|
| Prérequis VSCode, Python, Node, Git, UV, Docker Desktop | Environnement local |
| Ollama modèles `qwen3.5:0.8b` + `qwen3-embedding:0.6b` | `docker exec ollama ollama pull …` |
| Architecture : Front React + FastAPI + ES + Ollama embedding/rerank | `docker-compose.yml` |
| Script d’indexation Python | `index_data/main.py` |
| 1 index ES par cluster + recherche clusters puis BM25 | `index_data` + `fastapi-backend` |
| Rendu archive/repo par mail | Ce dépôt |

Théorie (RAG, harnais, GEO ChatGPT `prod_400_rerank400_cluster100_bm10_ann4096`) : slides de cours, pas de code à produire.

---

## Partie 2 — TD1

### TD1-1 Notebook Colab

| Prompt / étape | Fichier |
|----------------|---------|
| pCloud `getpublinkdownload` → path + host | `td1_colab/TD1_1_embeddings_clustering.ipynb` cellule 1 |
| embeddings Hugging Face | cellule 2 |
| TSV projector.tensorflow.org | cellule 3 |
| Visualiser PCA → UMAP | consignes dans le notebook |
| K-means sur `line_embeddings` | cellule 4 |
| TSV clusters + 3 premières lignes | cellule 5 |

### TD1-2 docker-compose

| Prompt | Fichier |
|--------|---------|
| `adds elasticsearch … 9.3.2 … cors … xpack.security.enabled false` | `docker-compose.yml` + `elasticsearch/elasticsearch.yml` |
| Vérifier http://localhost:9200 | OK |
| `adds ollama docker with gpu access` | `docker-compose.gpu.yml` (Linux+NVIDIA) ; défaut CPU macOS |
| Vérifier http://localhost:11434 | OK |
| `ollama pull qwen3-embedding:0.6b` | modèle présent |

### TD1-3 indexation

| Prompt | Fichier |
|--------|---------|
| `uv init index_data` | `index_data/` |
| `get_data_from_cloud` / `get_data_embeddings` / `cluster_data` | `index_data/main.py` |
| Embeddings via Ollama localhost:11434 | `get_data_embeddings` |
| Index `data_1`… + `data_clusters` | `index_to_elasticsearch` |

---

## Partie 3 — TD2

| TD | Prompt / étape | Fichier |
|----|----------------|---------|
| TD2-1 | `uv init fastapi-backend` + deps | `fastapi-backend/` |
| TD2-1 | search `data_clusters` puis `data_X` via `data_index` | `fastapi-backend/main.py` |
| TD2-1 | `add CORSMiddleware` | `main.py` |
| TD2-1 | Dockerfile uv python 3.12 trixie + `uv sync` + `uv run` | `fastapi-backend/Dockerfile` |
| TD2-1 | `adds fastapi backend built from folder fastapi-backend` | `docker-compose.yml` |
| TD2-2 | launch.json FastAPI debug | `.vscode/launch.json` |
| TD2-3 | `ELASTICSEARCH_HOST=http://elasticsearch:9200` | `docker-compose.yml` |
| TD2-3 | CMD debugpy listen 5678 + port `5678:5678` | Dockerfile + compose |
| TD2-3 | Remote Attach localhost:5678 | `.vscode/launch.json` |
| TD2-4 | `npm create vite@latest react-front` React TS | `react-front/` |
| TD2-5 | `write a Dockerfile for a vite project` + ports 3000:80 | `react-front/Dockerfile` |
| TD2-6 | input + bouton **Search** → `http://localhost:8000/search` | `App.tsx` |
| TD2-6 | KNN sur `data_clusters` | `search_clusters` |
| TD2-6 | recherche hybride BM25+KNN | `hybrid_search_in_cluster` |

---

## Partie 4 — TD3

| Prompt / étape | Fichier |
|----------------|---------|
| `ollama pull qwen3.5:0.8b` | modèle présent |
| Prompt rerank exact (JSON yes/no enum + logprobs + `p_yes/(p_yes+p_no)` + sort desc) | `fastapi-backend/rerank.py` |
| Schema objet `answer` enum yes/no | `JSON_SCHEMA` dans `rerank.py` |
| Appels parallèles query×document | `ThreadPoolExecutor` dans `rerank()` |

---

## Commandes de validation

```bash
cd iasearch
docker compose up -d --build
curl http://localhost:9200/
curl http://localhost:11434/
curl http://localhost:8000/docs
curl http://localhost:3000/
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"travaux de voirie","top_clusters":3,"top_docs":5,"use_rerank":true}'
```
