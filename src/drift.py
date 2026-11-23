import pandas as pd
from scipy.stats import ks_2samp
from .config import FEATURES

def check_drift(reference, incoming, alpha=0.05):
    results = []
    for feature in FEATURES:
        a = pd.to_numeric(reference[feature], errors="coerce").dropna()
        b = pd.to_numeric(incoming[feature], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            results.append({
                "feature": feature, "statistic": None, "p_value": None,
                "drift": False, "note": "Not enough valid observations"
            })
            continue
        stat, p = ks_2samp(a, b)
        results.append({
            "feature": feature, "statistic": float(stat), "p_value": float(p),
            "drift": bool(p < alpha), "note": "Distribution changed" if p < alpha else "No significant change detected"
        })
    flagged = sum(x["drift"] for x in results)
    return {
        "alpha": alpha,
        "features_checked": len(results),
        "features_flagged": flagged,
        "overall_drift_signal": flagged > 0,
        "results": results,
        "interpretation": "Drift is a review signal, not automatic proof of an attack."
    }
