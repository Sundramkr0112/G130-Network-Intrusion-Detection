from pathlib import Path
import numpy as np
import pandas as pd
from .config import FEATURES, TARGET

def load_dataset(path):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return clean_dataset(df)

def clean_dataset(df):
    df = df.copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET])
    # Median imputation is applied later using a training-fitted pipeline.
    return df

def split_features_target(df):
    return df[FEATURES], df[TARGET].astype(str)
