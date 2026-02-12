import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Ensure repo root is on sys.path when running via Streamlit.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.db import get_engine
from src.ml.load_features import RENAME
from src.ml.preprocess import get_feature_cols, make_preprocess
from src.ml.predict import (
    predict_single,
    resolve_model_uris,
    SEGMENT_NAME,
    SEGMENT_OFFER,
    batch_score,
    MODEL_ARTIFACT_PATH,
    VIZ_ARTIFACT_PATH,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


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

DATA_CSV_PATH = REPO_ROOT / "data" / "Customer Data.csv"


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


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
def load_local_data() -> pd.DataFrame:
    if not DATA_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_CSV_PATH}. "
            "Place 'Customer Data.csv' in the data/ folder."
        )
    df = pd.read_csv(DATA_CSV_PATH)
    df = df.rename(columns=RENAME)
    feature_cols = get_feature_cols()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")
    return df


@st.cache_resource
def train_local_models(df: pd.DataFrame, n_clusters: int = 4):
    preprocess = make_preprocess()
    seg_pipe = Pipeline([
        ("preprocess", preprocess),
        ("pca", PCA(n_components=0.9, random_state=42)),
        ("model", KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)),
    ])
    seg_pipe.fit(df[FEATURE_COLS])

    viz_pipe = Pipeline([
        ("preprocess", preprocess),
        ("pca", PCA(n_components=2, random_state=42)),
    ])
    viz_pipe.fit(df[FEATURE_COLS])
    return seg_pipe, viz_pipe


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
    import mlflow
    import mlflow.sklearn

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    seg_model = mlflow.sklearn.load_model(seg_uri)
    viz_model = mlflow.sklearn.load_model(viz_uri)
    return seg_model, viz_model


@st.cache_resource
def load_seg_model(seg_uri: str, tracking_uri: Optional[str]):
    import mlflow
    import mlflow.sklearn

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


def load_random_row_local(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {c: 0.0 for c in FEATURE_COLS}
    return df.sample(1).iloc[0][FEATURE_COLS].to_dict()


def load_sample_local(df: pd.DataFrame, limit: int = 500) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLS)
    size = min(limit, len(df))
    return df.sample(size)[FEATURE_COLS].copy()


def load_feature_stats_local(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if df.empty:
        return {}
    stats: Dict[str, Dict[str, float]] = {}
    for c in FEATURE_COLS:
        series = df[c].dropna()
        stats[c] = {
            "min": float(series.min()) if not series.empty else 0.0,
            "max": float(series.max()) if not series.empty else 0.0,
        }
    return stats


def load_row_count_local(df: pd.DataFrame) -> int:
    return int(len(df))


def init_session_state(db_uri: str, local_df: Optional[pd.DataFrame] = None):
    if "features" in st.session_state:
        return
    if local_df is not None:
        st.session_state["features"] = load_random_row_local(local_df)
    else:
        st.session_state["features"] = load_random_row(db_uri)


def set_features(values: Dict[str, float], update_inputs: bool = False) -> None:
    st.session_state["features"] = values
    if update_inputs:
        for name, value in values.items():
            key = f"input_{name}"
            if name in INT_FEATURES:
                st.session_state[key] = int(round(value))
            else:
                st.session_state[key] = float(value)
        st.rerun()


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
            key = f"input_{name}"
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
                    if key not in st.session_state:
                        st.session_state[key] = default_int
                    values[name] = st.number_input(
                        label=name,
                        step=1,
                        format="%d",
                        key=key,
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
                    if key not in st.session_state:
                        st.session_state[key] = default_val
                    values[name] = st.number_input(
                        label=name,
                        step=step,
                        format=fmt,
                        key=key,
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


def generate_cluster_profile_local(
    df: pd.DataFrame,
    seg_model,
    cluster_id: int,
) -> Dict[str, float]:
    sample_df = load_sample_local(df, limit=4000)
    if sample_df.empty:
        raise ValueError("No data available for demo mode.")
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
    stats = load_feature_stats_local(df)
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


def predict_single_local(
    features: Dict[str, float],
    seg_model,
    viz_model,
) -> Dict[str, object]:
    feature_cols = get_feature_cols()
    df = pd.DataFrame([features], columns=feature_cols)
    cluster_id = int(seg_model.predict(df)[0])
    pc1, pc2 = [float(x) for x in viz_model.transform(df)[0]]
    return {
        "cluster_id": cluster_id,
        "segment_name": SEGMENT_NAME.get(cluster_id, f"segment_{cluster_id}"),
        "offer": SEGMENT_OFFER.get(cluster_id, "generic offer"),
        "pc1": pc1,
        "pc2": pc2,
        "seg_uri": "local",
        "viz_uri": "local",
    }

default_demo = _env_truthy("DEMO_MODE") or _env_truthy("STREAMLIT_CLOUD")
if "demo_mode" not in st.session_state:
    st.session_state["demo_mode"] = default_demo

with st.sidebar:
    st.subheader("Mode")
    st.checkbox("Demo mode (no DB/MLflow)", key="demo_mode")
    st.caption("Demo mode trains locally from the CSV in data/ and skips DB/MLflow.")

demo_mode = bool(st.session_state["demo_mode"])

if not demo_mode:
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
    os.environ["DB_URI"] = st.session_state["db_uri"]
    init_session_state(st.session_state["db_uri"])

    tracking_default = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    tracking_uri = normalize_tracking_uri(tracking_default)
    if tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

    seg_uri = st.session_state["seg_uri"]
    viz_uri = st.session_state["viz_uri"]
else:
    try:
        local_df = load_local_data()
    except Exception as exc:
        st.error(f"Demo mode unavailable: {exc}")
        st.stop()
    init_session_state("demo", local_df=local_df)
    seg_model_local, viz_model_local = train_local_models(local_df)

with st.sidebar:
    st.subheader("Generate Profile")
    if st.button("Generate random user"):
        if demo_mode:
            set_features(load_random_row_local(local_df), update_inputs=True)
        else:
            set_features(load_random_row(st.session_state["db_uri"]), update_inputs=True)

    st.divider()
    cluster_options = {f"{k} — {SEGMENT_NAME.get(k, f'segment_{k}')}": k for k in sorted(SEGMENT_NAME)}
    cluster_label = st.selectbox("Cluster", list(cluster_options.keys()))
    if st.button("Generate by cluster"):
        if demo_mode:
            try:
                profile = generate_cluster_profile_local(
                    local_df,
                    seg_model_local,
                    cluster_options[cluster_label],
                )
                set_features(profile, update_inputs=True)
            except Exception as exc:
                st.warning(f"Cluster-based generation failed: {exc}")
        else:
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
                    set_features(profile, update_inputs=True)
                except Exception as exc:
                    st.warning(f"Cluster-based generation failed: {exc}")



with st.form("prediction_form"):
    st.subheader("Input Features")
    if demo_mode:
        stats = load_feature_stats_local(local_df)
    else:
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
        if demo_mode:
            result = predict_single_local(inputs, seg_model_local, viz_model_local)
        else:
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

        if demo_mode:
            row_count = load_row_count_local(local_df)
            limit = int(min(row_count, 3000)) if row_count else 3000
            sample_df = load_sample_local(local_df, limit=limit)
            if sample_df.empty:
                raise ValueError("No data in CSV to build PCA background.")
            seg_model, viz_model = seg_model_local, viz_model_local
        else:
            row_count = load_row_count(st.session_state["db_uri"])
            limit = 3000
            if row_count:
                limit = int(min(row_count, 3000))
            sample_df = load_sample(st.session_state["db_uri"], limit=limit)
            if sample_df.empty:
                raise ValueError("No data in customer_features to build PCA background.")
            seg_model, viz_model = load_models(result["seg_uri"], result["viz_uri"], tracking_uri)
        Z = viz_model.transform(sample_df[FEATURE_COLS])
        labels = seg_model.predict(sample_df[FEATURE_COLS])
        pc1, pc2 = result["pc1"], result["pc2"]

        fig, ax = plt.subplots(figsize=(6, 4))
        cmap = plt.cm.get_cmap("viridis", 4)
        for cluster_id in sorted(set(labels)):
            mask = labels == cluster_id
            ax.scatter(
                Z[mask, 0],
                Z[mask, 1],
                s=10,
                alpha=0.5,
                color=cmap(int(cluster_id)),
                label=f"cluster {int(cluster_id)}",
            )
        ax.scatter([pc1], [pc2], s=80, color="red", label="current")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best")
        plot_col = st.columns([1, 1, 1])[1]
        with plot_col:
            st.pyplot(fig, clear_figure=True, use_container_width=True)
    except Exception as exc:
        st.warning(f"PCA plot unavailable: {exc}")
