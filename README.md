# G130 - Explainable and Drift-Aware ML Framework for Network Intrusion Detection

A runnable academic prototype based on the GLA University G130 project synopsis (2026-27).

## What is included
- Random Forest network-flow classifier
- CSV upload and preprocessing pipeline
- Precision, recall, F1-score, confusion matrix and classification report
- Feature-importance explanations; optional SHAP explanations when installed
- Statistical drift checks using the two-sample Kolmogorov-Smirnov test
- Streamlit dashboard
- FastAPI prediction and drift endpoints
- Small synthetic demo dataset so the application can be tested without downloading a large benchmark dataset
- Tests and reproducible training script

> Important: the included CSV is synthetic demonstration data, not CIC-IDS2017 or UNSW-NB15.
> For research evaluation, obtain the official benchmark dataset, follow its licence/usage terms,
> map its label column to `Label`, and evaluate on a separated test set. Do not use this prototype
> as a production security system.

## Requirements
- Python 3.10 or newer
- 8 GB RAM recommended for larger CSV datasets

## Setup (Windows PowerShell)
```powershell
cd G130_Network_Intrusion_Detection
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the demo
1. Train the model:
```powershell
python -m src.train --data data/sample_network_flows.csv
```
2. Start the dashboard:
```powershell
streamlit run app.py
```
3. In another terminal, start the API:
```powershell
uvicorn api.main:app --reload
```
4. Open the dashboard at `http://localhost:8501` and API docs at `http://127.0.0.1:8000/docs`.

The model and metadata are written to `artifacts/`.

## API endpoints
- `GET /health` - service health
- `GET /model-info` - model metadata
- `POST /predict` - predict one network flow
- `POST /predict-batch` - predict multiple network flows
- `POST /drift-check` - compare reference and incoming CSV files

Example `/predict` request:
```json
{
  "flow_duration": 1200,
  "total_fwd_packets": 8,
  "total_backward_packets": 4,
  "flow_bytes_per_sec": 950.5,
  "flow_packets_per_sec": 12.0,
  "packet_length_mean": 420.0,
  "packet_length_std": 35.0,
  "syn_flag_count": 1,
  "ack_flag_count": 1,
  "destination_port": 443
}
```

## Dataset format
The demo uses these numeric features:
`flow_duration`, `total_fwd_packets`, `total_backward_packets`,
`flow_bytes_per_sec`, `flow_packets_per_sec`, `packet_length_mean`,
`packet_length_std`, `syn_flag_count`, `ack_flag_count`, `destination_port`

Target column: `Label` (`BENIGN` or an attack label).

For CIC-IDS2017 / UNSW-NB15, create a clean CSV with numeric flow features and a `Label` column.
Remove identifiers and leakage columns, handle infinite values, and document all preprocessing decisions.
Do not combine train and test records before splitting.

## Project workflow
1. Load and validate network-flow CSV data.
2. Clean invalid/infinite/missing numeric values.
3. Split data with stratification when possible.
4. Train a Random Forest baseline.
5. Evaluate with precision, recall, F1 and confusion matrix.
6. Explain influential features using model importance (and optional SHAP).
7. Compare reference and incoming feature distributions for drift.
8. Display predictions, explanations, drift warnings and evaluation metrics.

## Limitations
- Drift is a review signal, not proof of an attack.
- Synthetic demo data is only for checking that the software runs.
- Feature importance is a global explanation; it is not a guaranteed causal explanation for an individual event.
- Dataset-specific column mapping and careful validation are required before reporting benchmark results.
