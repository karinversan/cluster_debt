import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import mlflow
import mlflow.sklearn

# Ensure repo root is on sys.path when running via Streamlit.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.db import get_engine
from src.ml.preprocess import get_feature_cols
from src.ml.predict import (
    predict_single,
    resolve_model_uris,
    SEGMENT_NAME,
    batch_score,
    MODEL_ARTIFACT_PATH,
    VIZ_ARTIFACT_PATH,
)
from src.ml.train import train


st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation Demo")

FEATURE_COLS = get_feature_cols()

AMOUNT_FEATURES = [
    "balance", "purchases", "oneoff_purchases",
    "installments_purchases", "cash_advance",
    "payments", "credit_limit",
]
COUNT_FEATURES = ["purchases_trx", "cash_advance_trx"]
FREQUENCY_FEATURES = [
    "balance_frequency", "purchases_frequency",
    "oneoff_purchases_frequency",
    "purchases_installments_frequency",
    "cash_advance_frequency",
    "prc_full_payment",
]
MINPAY_FEATURES = ["minimum_payments"]
TENURE_FEATURES = ["tenure"]
INT_FEATURES = COUNT_FEATURES + TENURE_FEATURES


def _clamp(value: float, min_v: Optional[float], max_v: Optional[float]) -> float:
    if min_v is not None:
        value = max(value, min_v)
    if max_v is not None:
        value = min(value, max_v)
    return value


def normalize_db_uri(db_uri: str) -> str:
    if "@localhost" in db_uri:
        return db_uri.replace("@localhost", "@127.0.0.1")
    if "@127.0.0.1:5432" in db_uri:
        return db_uri.replace("@127.0.0.1:5432", "@127.0.0.1:5433")
    return db_uri


def is_valid_mlflow_uri(uri: str) -> bool:
    return uri.startswith("runs:/") or uri.startswith("models:/") or "://" in uri


def normalize_tracking_uri(uri: str) -> str:
    if uri.startswith("http://localhost:"):
        return uri.replace("http://localhost:", "http://127.0.0.1:")
    return uri


@st.cache_resource
def get_engine_cached(db_uri: str):
    return get_engine(db_uri)


@st.cache_data
def load_random_row(db_uri: str) -> Dict[str, float]:
    try:
        engine = get_engine_cached(db_uri)
        cols = ", ".join(FEATURE_COLS)
        df = pd.read_sql(f"SELECT {cols} FROM customer_features ORDER BY random() LIMIT 1", engine)
        if df.empty:
            return {c: 0.0 for c in FEATURE_COLS}
        return df.iloc[0].to_dict()
    except Exception:
        return {c: 0.0 for c in FEATURE_COLS}


@st.cache_data
def load_sample(db_uri: str, limit: int = 500) -> pd.DataFrame:
    try:
        engine = get_engine_cached(db_uri)
        cols = ", ".join(FEATURE_COLS)
        return pd.read_sql(
            f"SELECT {cols} FROM customer_features ORDER BY random() LIMIT {limit}",
            engine,
        )
    except Exception:
        return pd.DataFrame(columns=FEATURE_COLS)


@st.cache_resource
def load_models(seg_uri: str, viz_uri: str, tracking_uri: Optional[str]):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    seg_model = mlflow.sklearn.load_model(seg_uri)
    viz_model = mlflow.sklearn.load_model(viz_uri)
    return seg_model, viz_model


@st.cache_resource
def load_seg_model(seg_uri: str, tracking_uri: Optional[str]):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return mlflow.sklearn.load_model(seg_uri)


@st.cache_data
def load_feature_stats(db_uri: str) -> Dict[str, Dict[str, float]]:
    try:
        engine = get_engine_cached(db_uri)
        cols = []
        for c in FEATURE_COLS:
            cols.append(f"MIN({c}) AS min_{c}")
            cols.append(f"MAX({c}) AS max_{c}")
        df = pd.read_sql(f"SELECT {', '.join(cols)} FROM customer_features", engine)
        if df.empty:
            return {}
        row = df.iloc[0].to_dict()
        stats: Dict[str, Dict[str, float]] = {}
        for c in FEATURE_COLS:
            stats[c] = {
                "min": float(row.get(f"min_{c}") or 0.0),
                "max": float(row.get(f"max_{c}") or 0.0),
            }
        return stats
    except Exception:
        return {}


@st.cache_data
def load_row_count(db_uri: str) -> Optional[int]:
    try:
        engine = get_engine_cached(db_uri)
        df = pd.read_sql("SELECT COUNT(*) AS cnt FROM customer_features", engine)
        return int(df.iloc[0]["cnt"])
    except Exception:
        return None


def init_session_state(db_uri: str):
    if "features" in st.session_state:
        return
    st.session_state["features"] = load_random_row(db_uri)


def set_features(values: Dict[str, float]) -> None:
    st.session_state["features"] = values


def render_group(
    title: str,
    features: List[str],
    stats: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    values: Dict[str, float] = {}
    with st.expander(title, expanded=True):
        cols = st.columns(3)
        for idx, name in enumerate(features):
            default = float(st.session_state["features"].get(name, 0.0) or 0.0)
            stat = stats.get(name, {})
            min_v = stat.get("min")
            max_v = stat.get("max")
            with cols[idx % 3]:
                if name in FREQUENCY_FEATURES:
                    min_v = 0.0
                    max_v = 1.0
                elif name in INT_FEATURES:
                    min_v = 0.0 if min_v is None else max(min_v, 0.0)

                if name in INT_FEATURES:
                    default_int = int(round(_clamp(default, min_v, max_v)))
                    kwargs = {"min_value": int(min_v or 0)}
                    if max_v is not None:
                        kwargs["max_value"] = int(max_v)
                    help_text = None
                    if max_v is not None:
                        help_text = f"Integer range: {int(min_v or 0)}–{int(max_v)}"
                    values[name] = st.number_input(
                        label=name,
                        value=default_int,
                        step=1,
                        format="%d",
                        key=f"input_{name}",
                        help=help_text,
                        **kwargs,
                    )
                else:
                    default_val = float(_clamp(default, min_v, max_v))
                    kwargs = {}
                    if min_v is not None:
                        kwargs["min_value"] = float(min_v)
                    if max_v is not None:
                        kwargs["max_value"] = float(max_v)
                    step = 0.01
                    fmt = "%.4f" if name in FREQUENCY_FEATURES else "%.2f"
                    help_text = None
                    if min_v is not None or max_v is not None:
                        help_text = f"Range: {min_v if min_v is not None else '-'}–{max_v if max_v is not None else '-'}"
                    values[name] = st.number_input(
                        label=name,
                        value=default_val,
                        step=step,
                        format=fmt,
                        key=f"input_{name}",
                        help=help_text,
                        **kwargs,
                    )
    return values


def generate_cluster_profile(
    seg_uri: str,
    cluster_id: int,
    tracking_uri: Optional[str],
    db_uri: str,
) -> Dict[str, float]:
    sample_df = load_sample(db_uri, limit=4000)
    if sample_df.empty:
        raise ValueError("No data available in customer_features.")
    seg_model = load_seg_model(seg_uri, tracking_uri)
    labels = seg_model.predict(sample_df[FEATURE_COLS])
    sample_df = sample_df.copy()
    sample_df["cluster_id"] = labels
    cluster_df = sample_df[sample_df["cluster_id"] == cluster_id]
    if cluster_df.empty:
        cluster_df = sample_df
    rng = np.random.default_rng()
    profile: Dict[str, float] = {}
    for name in FEATURE_COLS:
        series = cluster_df[name].dropna()
        if series.empty:
            profile[name] = 0.0
            continue
        if name in INT_FEATURES:
            value = float(series.sample(1).iloc[0])
        elif name in FREQUENCY_FEATURES:
            mean = float(series.mean())
            std = float(series.std() or 0.0)
            value = mean + rng.normal(0.0, std if std > 0 else 0.0)
        else:
            if (series > 0).all():
                log_vals = np.log1p(series.values)
                mu = float(np.mean(log_vals))
                sigma = float(np.std(log_vals))
                value = float(np.expm1(rng.normal(mu, sigma if sigma > 0 else 0.0)))
            else:
                mean = float(series.mean())
                std = float(series.std() or 0.0)
                value = mean + rng.normal(0.0, std if std > 0 else 0.0)
        profile[name] = float(value)
    stats = load_feature_stats(db_uri)
    for name in INT_FEATURES:
        if name in profile:
            profile[name] = int(round(profile[name]))
    for name, value in profile.items():
        bounds = stats.get(name, {})
        min_v = bounds.get("min")
        max_v = bounds.get("max")
        if name in FREQUENCY_FEATURES:
            min_v = 0.0
            max_v = 1.0
        profile[name] = float(_clamp(float(value), min_v, max_v))
    return profile

if "db_uri" not in st.session_state:
    st.session_state["db_uri"] = os.environ.get(
        "DB_URI",
        "postgresql+psycopg2://mlops:mlops@127.0.0.1:5433/segmentation",
    )
if "seg_uri" not in st.session_state:
    st.session_state["seg_uri"] = os.environ.get("SEG_MODEL_URI", "")
if "viz_uri" not in st.session_state:
    st.session_state["viz_uri"] = os.environ.get("VIZ_MODEL_URI", "")
if st.session_state["seg_uri"] and not is_valid_mlflow_uri(st.session_state["seg_uri"]):
    st.session_state["seg_uri"] = ""
if st.session_state["viz_uri"] and not is_valid_mlflow_uri(st.session_state["viz_uri"]):
    st.session_state["viz_uri"] = ""

st.session_state["db_uri"] = normalize_db_uri(st.session_state["db_uri"])
init_session_state(st.session_state["db_uri"])

with st.sidebar:
    st.subheader("Model URIs")
    st.session_state["db_uri"] = normalize_db_uri(
        st.text_input("DB_URI", value=st.session_state["db_uri"])
    )
    if "@127.0.0.1:5433" not in st.session_state["db_uri"]:
        st.caption("Tip: Docker Postgres is mapped to 127.0.0.1:5433.")
        if st.button("Use docker DB (5433)"):
            st.session_state["db_uri"] = "postgresql+psycopg2://mlops:mlops@127.0.0.1:5433/segmentation"
    os.environ["DB_URI"] = st.session_state["db_uri"]
    tracking_default = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    tracking_uri = normalize_tracking_uri(st.text_input("MLFLOW_TRACKING_URI", value=tracking_default))
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

    if st.button("Resolve latest from MLflow"):
        try:
            seg_uri_default, viz_uri_default = resolve_model_uris()
            st.session_state["seg_uri"] = seg_uri_default
            st.session_state["viz_uri"] = viz_uri_default
        except Exception as exc:
            st.warning(f"Failed to resolve latest MLflow run: {exc}")

    seg_uri = st.text_input("SEG_MODEL_URI", value=st.session_state["seg_uri"])
    viz_uri = st.text_input("VIZ_MODEL_URI", value=st.session_state["viz_uri"])

    if st.button("Generate random sample"):
        set_features(load_random_row(st.session_state["db_uri"]))

    row_count = load_row_count(st.session_state["db_uri"])
    if row_count is None:
        st.warning("DB connection failed. Check DB_URI.")
    else:
        st.caption(f"Rows in customer_features: {row_count}")

    cluster_options = {f"{k} — {SEGMENT_NAME.get(k, f'segment_{k}')}": k for k in sorted(SEGMENT_NAME)}
    cluster_label = st.selectbox("Generate by cluster", list(cluster_options.keys()))
    if st.button("Generate cluster-based sample"):
        if not seg_uri or not is_valid_mlflow_uri(seg_uri):
            try:
                seg_uri_default, viz_uri_default = resolve_model_uris()
                st.session_state["seg_uri"] = seg_uri_default
                st.session_state["viz_uri"] = viz_uri_default
                seg_uri = seg_uri_default
            except Exception as exc:
                st.warning(f"SEG_MODEL_URI is empty and resolve failed: {exc}")
        if seg_uri:
            try:
                profile = generate_cluster_profile(
                    seg_uri,
                    cluster_options[cluster_label],
                    tracking_uri,
                    st.session_state["db_uri"],
                )
                set_features(profile)
            except Exception as exc:
                st.warning(f"Cluster-based generation failed: {exc}")

    st.divider()
    st.subheader("Train / Refresh")
    use_pca = st.checkbox("Use PCA", value=True)
    pca_var = st.slider("PCA variance", min_value=0.6, max_value=0.99, value=0.9, step=0.01)
    if st.button("Train / Refresh model"):
        try:
            os.environ["DB_URI"] = st.session_state["db_uri"]
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
            with st.spinner("Training model..."):
                run_id = train(k=4, use_pca=use_pca, pca_var=float(pca_var))
            st.session_state["seg_uri"] = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"
            st.session_state["viz_uri"] = f"runs:/{run_id}/{VIZ_ARTIFACT_PATH}"
            with st.spinner("Batch scoring..."):
                batch_score(run_id=run_id, truncate=True)
            st.success(f"Training complete. Run: {run_id}")
        except Exception as exc:
            st.error(f"Train/refresh failed: {exc}")


with st.form("prediction_form"):
    st.subheader("Input Features")
    stats = load_feature_stats(st.session_state["db_uri"])
    inputs: Dict[str, float] = {}
    inputs.update(render_group("Amount Features", AMOUNT_FEATURES, stats))
    inputs.update(render_group("Count Features", COUNT_FEATURES, stats))
    inputs.update(render_group("Frequency Features", FREQUENCY_FEATURES, stats))
    inputs.update(render_group("Minimum Payments", MINPAY_FEATURES, stats))
    inputs.update(render_group("Tenure", TENURE_FEATURES, stats))

    submitted = st.form_submit_button("Predict cluster")


if submitted:
    # Persist latest inputs
    set_features(inputs)

    try:
        if not seg_uri or not is_valid_mlflow_uri(seg_uri):
            seg_uri, viz_uri = resolve_model_uris()
            st.session_state["seg_uri"] = seg_uri
            st.session_state["viz_uri"] = viz_uri
        result = predict_single(inputs, seg_uri=seg_uri, viz_uri=viz_uri)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.success(f"Cluster: {result['cluster_id']} — {result['segment_name']}")
    st.info(f"Offer: {result['offer']}")

    st.subheader("PCA Visualization")
    try:
        import matplotlib.pyplot as plt

        sample_df = load_sample(st.session_state["db_uri"], limit=600)
        if sample_df.empty:
            raise ValueError("No data in customer_features to build PCA background.")
        _, viz_model = load_models(result["seg_uri"], result["viz_uri"], tracking_uri)
        Z = viz_model.transform(sample_df[FEATURE_COLS])
        pc1, pc2 = result["pc1"], result["pc2"]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(Z[:, 0], Z[:, 1], s=10, alpha=0.4, label="dataset")
        ax.scatter([pc1], [pc2], s=80, color="red", label="current")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best")
        st.pyplot(fig, clear_figure=True)
    except Exception as exc:
        st.warning(f"PCA plot unavailable: {exc}")
