import os
from typing import Dict, Tuple, Optional

import pandas as pd

from src.ml.db import get_engine
from src.ml.preprocess import get_feature_cols, validate_input

EXPERIMENT_NAME = "customer-segmentation"

SEGMENT_NAME = {
    0: "Active Revolving",
    1: "Active Transactors",
    2: "Low Activity",
    3: "Cash Advance / High Risk",
}

SEGMENT_OFFER = {
    0: "Balance transfer / refinancing",
    1: "Rewards / premium card",
    2: "Activation bonus",
    3: "Debt consolidation / limit cash advance",
}

MODEL_ARTIFACT_PATH = "segmentation_model"
VIZ_ARTIFACT_PATH = "viz_pca_model"


def _is_valid_uri(uri: Optional[str]) -> bool:
    if not uri:
        return False
    return uri.startswith("runs:/") or uri.startswith("models:/") or "://" in uri


def _latest_run_id() -> str:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found in MLflow.")
    runs = client.search_runs(
        [exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs found in experiment '{EXPERIMENT_NAME}'.")
    return runs[0].info.run_id


def _extract_run_id(uri: Optional[str]) -> Optional[str]:
    if not uri:
        return None
    if uri.startswith("runs:/"):
        tail = uri.split("runs:/", 1)[1]
        return tail.split("/", 1)[0] if "/" in tail else tail
    return None


def resolve_model_uris(
    seg_uri: Optional[str] = None,
    viz_uri: Optional[str] = None,
) -> Tuple[str, str]:
    seg_uri = seg_uri or os.environ.get("SEG_MODEL_URI")
    viz_uri = viz_uri or os.environ.get("VIZ_MODEL_URI")
    if _is_valid_uri(seg_uri) and _is_valid_uri(viz_uri):
        return seg_uri, viz_uri
    if not _is_valid_uri(seg_uri):
        seg_uri = None
    if not _is_valid_uri(viz_uri):
        viz_uri = None

    run_id = _latest_run_id()
    return (
        seg_uri or f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}",
        viz_uri or f"runs:/{run_id}/{VIZ_ARTIFACT_PATH}",
    )


def predict_single(
    features: Dict[str, float],
    seg_uri: Optional[str] = None,
    viz_uri: Optional[str] = None,
) -> Dict[str, object]:
    import mlflow
    import mlflow.sklearn

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    feature_cols = get_feature_cols()
    df = pd.DataFrame([features], columns=feature_cols)
    validate_input(df)

    seg_uri, viz_uri = resolve_model_uris(seg_uri, viz_uri)

    seg_model = mlflow.sklearn.load_model(seg_uri)
    viz_model = mlflow.sklearn.load_model(viz_uri)

    cluster_id = int(seg_model.predict(df)[0])
    pc1, pc2 = [float(x) for x in viz_model.transform(df)[0]]

    return {
        "cluster_id": cluster_id,
        "segment_name": SEGMENT_NAME.get(cluster_id, f"segment_{cluster_id}"),
        "offer": SEGMENT_OFFER.get(cluster_id, "generic offer"),
        "pc1": pc1,
        "pc2": pc2,
        "seg_uri": seg_uri,
        "viz_uri": viz_uri,
    }


def batch_score(
    run_id: Optional[str] = None,
    seg_uri: Optional[str] = None,
    viz_uri: Optional[str] = None,
    truncate: bool = True,
    limit: Optional[int] = None,
) -> int:
    import mlflow
    import mlflow.sklearn

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if run_id:
        seg_uri = seg_uri or f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"
        viz_uri = viz_uri or f"runs:/{run_id}/{VIZ_ARTIFACT_PATH}"

    seg_uri, viz_uri = resolve_model_uris(seg_uri, viz_uri)
    run_id = run_id or _extract_run_id(seg_uri) or _latest_run_id()

    engine = get_engine()
    feature_cols = get_feature_cols()
    cols = ["customer_id", "event_timestamp"] + feature_cols
    select_cols = ", ".join(cols)
    query = f"SELECT {select_cols} FROM customer_features"
    if limit:
        query += f" LIMIT {int(limit)}"
    df = pd.read_sql(query, engine)
    if df.empty:
        return 0

    validate_input(df)
    seg_model = mlflow.sklearn.load_model(seg_uri)
    labels = seg_model.predict(df[feature_cols])

    seg_ids = pd.Series(labels).astype(int)
    seg_names = seg_ids.map(lambda x: SEGMENT_NAME.get(int(x), f"segment_{int(x)}"))

    result = pd.DataFrame({
        "customer_id": df["customer_id"].astype(str),
        "event_timestamp": df["event_timestamp"],
        "segment_id": seg_ids,
        "segment_name": seg_names,
        "model_name": MODEL_ARTIFACT_PATH,
        "model_version": run_id,
        "run_id": run_id,
    })

    with engine.begin() as conn:
        if truncate:
            conn.exec_driver_sql("TRUNCATE TABLE customer_segments;")
    result.to_sql("customer_segments", engine, if_exists="append", index=False, chunksize=2000)
    return int(len(result))
