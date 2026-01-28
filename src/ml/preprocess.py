import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer

AMOUNT_FEATURES = [
    "balance", "purchases", "oneoff_purchases",
    "installments_purchases", "cash_advance",
    "payments", "credit_limit",
]

COUNT_FEATURES = [
    "purchases_trx", "cash_advance_trx",
]

FREQUENCY_FEATURES = [
    "balance_frequency", "purchases_frequency",
    "oneoff_purchases_frequency",
    "purchases_installments_frequency",
    "cash_advance_frequency",
    "prc_full_payment",
]

MINPAY_FEATURE = ["minimum_payments"]
DROP_FEATURES = ["tenure"]

FEATURE_COLS = (
    AMOUNT_FEATURES
    + COUNT_FEATURES
    + FREQUENCY_FEATURES
    + MINPAY_FEATURE
    + DROP_FEATURES
)


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def make_preprocess() -> ColumnTransformer:
    log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")

    preprocess = ColumnTransformer(
        transformers=[
            ("amount", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("log", log1p),
                ("scale", RobustScaler()),
            ]), AMOUNT_FEATURES),
            ("count", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("log", log1p),
                ("scale", RobustScaler()),
            ]), COUNT_FEATURES),
            ("freq", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", RobustScaler()),
            ]), FREQUENCY_FEATURES),
            ("minpay", Pipeline([
                ("impute0_flag", SimpleImputer(
                    strategy="constant", fill_value=0, add_indicator=True
                )),
                ("log", log1p),
                ("scale", RobustScaler()),
            ]), MINPAY_FEATURE),
            ("drop", "drop", DROP_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocess


def get_feature_cols():
    return FEATURE_COLS
