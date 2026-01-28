import os
import tempfile

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.ml.db import get_engine
from src.ml.preprocess import make_preprocess, validate_input, get_feature_cols

EXPERIMENT_NAME = "customer-segmentation"
MODEL_NAME = "segmentation"
VIZ_MODEL_NAME = "segmentation_viz_pca"


def train(k: int = 4, use_pca: bool = True, pca_var: float = 0.9) -> str:
    import mlflow
    import mlflow.sklearn

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    engine = get_engine()
    df = pd.read_sql("SELECT * FROM customer_features", engine)

    feature_cols = get_feature_cols()
    X = df[feature_cols].copy()
    validate_input(X)

    preprocess = make_preprocess()

    if use_pca:
        seg_pipe = Pipeline([
            ("preprocess", preprocess),
            ("pca", PCA(n_components=pca_var, random_state=42)),
            ("model", KMeans(n_clusters=k, n_init="auto", random_state=42)),
        ])
    else:
        seg_pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", KMeans(n_clusters=k, n_init="auto", random_state=42)),
        ])

    labels = seg_pipe.fit_predict(X)

    X_emb = seg_pipe[:-1].transform(X)
    sil = silhouette_score(X_emb, labels)
    dbi = davies_bouldin_score(X_emb, labels)
    chi = calinski_harabasz_score(X_emb, labels)

    viz_pipe = Pipeline([
        ("preprocess", preprocess),
        ("pca", PCA(n_components=2, random_state=42)),
    ])
    viz_pipe.fit(X)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.log_param("n_clusters", k)
        mlflow.log_param("use_pca", use_pca)
        if use_pca:
            mlflow.log_param("pca_var", pca_var)

        mlflow.log_metric("silhouette", float(sil))
        mlflow.log_metric("davies_bouldin", float(dbi))
        mlflow.log_metric("calinski_harabasz", float(chi))

        shares = (
            pd.Series(labels)
            .value_counts(normalize=True)
            .sort_index()
            .rename("share")
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/cluster_shares.csv"
            shares.to_csv(path)
            mlflow.log_artifact(path)

        mlflow.sklearn.log_model(seg_pipe, artifact_path="segmentation_model")
        mlflow.sklearn.log_model(viz_pipe, artifact_path="viz_pca_model")

        return run.info.run_id


if __name__ == "__main__":
    print(train())
