from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    labels = sorted(set(y_test) | set(pred))
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0, output_dict=True)
    }
