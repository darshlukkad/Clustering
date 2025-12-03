# Clustering & Anomaly Detection Colabs

This folder contains one Colab-style notebook per assignment covering clustering and anomaly detection techniques:

- `KMeans_from_scratch.ipynb` — K-Means implemented from scratch with evaluation (elbow, silhouette) and comparison to scikit-learn.
- `Hierarchical_Clustering.ipynb` — Agglomerative / hierarchical clustering with dendrograms and cophenetic evaluation.
- `GaussianMixture_Clustering.ipynb` — Gaussian Mixture Models, AIC/BIC model selection, visualization.
- `DBSCAN_PyCaret.ipynb` — DBSCAN via PyCaret (with guarded import and usage notes for Colab).
- `Anomaly_pyOD.ipynb` — Anomaly detection examples using pyOD detectors (IForest, LOF, HBOS, AutoEncoder).
- `TimeSeries_Clustering.ipynb` — Time-series clustering (tslearn + DTW) with fallback notes for pretrained encoders.
- `Document_Clustering_LLM_Embeddings.ipynb` — Document clustering using sentence-transformers embeddings and HDBSCAN/UMAP.
- `Image_Clustering_ImageBind_or_CLIP.ipynb` — Image clustering using ImageBind (if available) or CLIP fallback.
- `Audio_Clustering_OpenL3_or_MFCC.ipynb` — Audio embeddings via openl3 (or MFCC fallback) and clustering.

Quick start (Colab)
1. Upload this folder to your Google Drive or open each notebook directly in Colab via "File > Open notebook > GitHub/Upload".
2. For each notebook: ensure required packages are installed. Many notebooks include an "Install notes" cell near the top — uncomment and run it.

Recommended minimal installs (uncomment in Colab):

```bash
# core
pip install numpy pandas scikit-learn matplotlib seaborn joblib umap-learn hdbscan

# optional / heavy (run only if you need them)
pip install sentence-transformers faiss-cpu librosa openl3 tslearn pyod

# PyCaret (very large; use only if you need PyCaret features)
pip install pycaret[full]
```

Colab runtime suggestions
- CPU runtime is enough for most demos. Use a GPU runtime for heavy embedding models (CLIP/ImageBind/OpenL3) to accelerate encoding.
- If installing PyCaret or other heavy packages, use a fresh Colab runtime and restart the runtime after installs.

Saving outputs
- Notebooks save simple artifacts locally (e.g., `*.pkl` or `*_clusters.csv`). To persist to Google Drive, mount Drive at the top of the notebook and write artifacts to `/content/drive/MyDrive/...`.

Notes
- Several notebooks include fallback code paths so they run even if a heavy library is unavailable (e.g., CLIP fallback to color histograms, openl3 fallback to MFCCs).
- If you want, I can prepare a `requirements.txt` with pinned versions and a lightweight GitHub README with links to Colab shareable URLs.

If you want me to run a full automated validation pass (execute each notebook headless), I can provide a list of commands and attempt to run them locally, but execution in this environment may be limited. Let me know if you'd like that next.
