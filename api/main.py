from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from src.config import FEATURES
from src.drift import check_drift

app = FastAPI(
    title="G130 Network Intrusion Detection API",
    description="Explainable and drift-aware academic prototype",
    version="1.0.0"
)
MODEL_PATH = Path("artifacts/intrusion_model.joblib")
META_PATH = Path("artifacts/model_metadata.json")

class Flow(BaseModel):
    flow_duration: float
    total_fwd_packets: float
    total_backward_packets: float
    flow_bytes_per_sec: float
    flow_packets_per_sec: float
    packet_length_mean: float
    packet_length_std: float
    syn_flag_count: float
    ack_flag_count: float
    destination_port: float

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Model not trained. Run the training command first.")
    return joblib.load(MODEL_PATH)

@app.get("/model-info")
def model_info():
    if not META_PATH.exists():
        raise HTTPException(404, "Model metadata not found. Train the model first.")
    import json
    return json.loads(META_PATH.read_text())

@app.post("/predict")
def predict(flow: Flow):
    model = get_model()
    row = pd.DataFrame([flow.model_dump()], columns=FEATURES)
    prediction = str(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    classes = model.named_steps["classifier"].classes_
    return {
        "prediction": prediction,
        "risk_label": "BENIGN" if prediction.upper() == "BENIGN" else "POTENTIAL_INTRUSION",
        "probabilities": {str(c): float(p) for c, p in zip(classes, probabilities)}
    }

@app.post("/predict-batch")
def predict_batch(flows: List[Flow]):
    if not flows:
        raise HTTPException(400, "Provide at least one flow.")
    model = get_model()
    frame = pd.DataFrame([x.model_dump() for x in flows], columns=FEATURES)
    preds = model.predict(frame)
    probs = model.predict_proba(frame)
    classes = model.named_steps["classifier"].classes_
    return {"predictions": [
        {"prediction": str(p), "probabilities": {str(c): float(v) for c, v in zip(classes, pr)}}
        for p, pr in zip(preds, probs)
    ]}

@app.post("/drift-check")
def drift_check(payload: Dict[str, str]):
    ref_path, new_path = payload.get("reference_csv"), payload.get("incoming_csv")
    if not ref_path or not new_path:
        raise HTTPException(400, "Provide reference_csv and incoming_csv paths.")
    try:
        ref, new = pd.read_csv(ref_path), pd.read_csv(new_path)
        missing = [f for f in FEATURES if f not in ref.columns or f not in new.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return check_drift(ref, new)
    except Exception as exc:
        raise HTTPException(400, str(exc))
