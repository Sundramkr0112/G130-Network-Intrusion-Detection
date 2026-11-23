from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from .config import FEATURES, RANDOM_STATE

def build_model():
    prep = ColumnTransformer(
        [("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), FEATURES)],
        remainder="drop"
    )
    classifier = RandomForestClassifier(
        n_estimators=250,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    return Pipeline([("preprocessing", prep), ("classifier", classifier)])
