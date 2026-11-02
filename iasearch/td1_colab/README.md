# TD1-1 — Notebook Colab + TensorFlow Projector

## Exécuté

Les 5 cellules du cours ont été exécutées (script local équivalent Colab) :

1. Download pCloud → 998 lignes
2. Embeddings Hugging Face (`paraphrase-multilingual-MiniLM-L12-v2`)
3. TSV lines → `projector_lines/`
4. KMeans k=10
5. TSV clusters → `projector_clusters/`

Notebook : `TD1_1_embeddings_clustering.ipynb` (à ouvrir aussi dans Google Colab).

## Étape manuelle demandée par le prof (Projector)

1. Ouvrir https://projector.tensorflow.org/
2. **Load** → choisir `projector_lines/vectors.tsv` puis `projector_lines/metadata.tsv`
3. Passer de **PCA** à **UMAP**
4. Recommencer avec `projector_clusters/vectors.tsv` + `metadata.tsv`
