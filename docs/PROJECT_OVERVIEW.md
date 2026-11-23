# Project overview and academic workflow

## Problem
Network traffic changes over time. A classifier trained on older flow records may become less reliable when traffic distributions change. Security analysts also need understandable reasons behind model predictions.

## Proposed solution
The prototype combines:
1. Supervised classification using a Random Forest baseline.
2. Feature-level explanations through model feature importance, with SHAP listed as an optional extension.
3. Statistical drift checks comparing reference and incoming feature distributions.
4. A dashboard and API for predictions, explanations, alerts and evaluation summaries.

## Experimental protocol
- Use official CIC-IDS2017 records as the primary benchmark.
- Use UNSW-NB15 only where feasible and document any schema mapping.
- Split data into train and held-out test sets before fitting preprocessing.
- Report class-wise precision, recall, F1-score, confusion matrix and macro averages.
- Record the dataset version, selected features, label mapping, split seed and software versions.
- For drift experiments, define a reference period and one or more later/incoming periods.
- Compare metrics before and after drift; retraining should be a separately documented experiment.

## Safety and interpretation
Use only public datasets and controlled, authorised lab traffic. Drift alerts are review signals, not proof of an attack. This research prototype is not a replacement for a production security team.
