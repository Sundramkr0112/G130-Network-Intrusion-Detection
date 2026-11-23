import pandas as pd
from src.drift import check_drift
from src.config import FEATURES

def test_drift_report_has_all_features():
    a = pd.DataFrame({f: [1, 2, 3, 4, 5] for f in FEATURES})
    b = pd.DataFrame({f: [100, 200, 300, 400, 500] for f in FEATURES})
    report = check_drift(a, b)
    assert report["features_checked"] == len(FEATURES)
    assert "overall_drift_signal" in report

def test_drift_interpretation_is_cautious():
    a = pd.DataFrame({f: [1, 2, 3, 4, 5] for f in FEATURES})
    report = check_drift(a, a)
    assert "not automatic proof" in report["interpretation"]
