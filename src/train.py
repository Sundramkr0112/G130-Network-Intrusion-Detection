import argparse, json
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from .data import load_dataset, split_features_target
from .model import build_model
from .evaluation import evaluate_model

def train(data_path, output_dir="artifacts"):
    df = load_dataset(data_path)
    X, y = split_features_target(df)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )
    model = build_model()
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "intrusion_model.joblib")
    metadata = {
        "model": "RandomForestClassifier",
        "rows": int(len(df)),
        "features": list(X.columns),
        "classes": sorted(y.unique().tolist()),
        "metrics": metrics
    }
    (out / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    result = train(args.data, args.output)
    print(json.dumps(result["metrics"], indent=2))
