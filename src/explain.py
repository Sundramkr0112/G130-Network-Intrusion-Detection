import pandas as pd
from .config import FEATURES

def feature_importance(model):
    clf = model.named_steps["classifier"]
    values = clf.feature_importances_
    return pd.DataFrame({"feature": FEATURES, "importance": values}).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)

def explain_one(model, row):
    # Transparent local fallback: show the strongest global model features.
    # This is not a causal or exact per-record explanation.
    return feature_importance(model).head(5).to_dict(orient="records")

def shap_available():
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False
