import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.decomposition import PCA

log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
drop_features = ["TENURE"]

preprocess = ColumnTransformer(
    transformers=[
        ("amount", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log", log1p),
            ("scale", RobustScaler())
        ]), amount_features),

        ("count", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log", log1p),
            ("scale", RobustScaler())
        ]), count_features),

        ("freq", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler())
        ]), frequency_features),

        ("minpay", Pipeline([
            ("impute0_flag", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True)),
            ("log", log1p),
            ("scale", RobustScaler())
        ]), minpay_feature),

        ("drop", "drop", drop_features),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

preprocess_with_pca = Pipeline([
    ('preprocess', preprocess),
    ('pca', PCA(n_components=0.9, random_state=42)) # сохраняем 90% диспресиии
])