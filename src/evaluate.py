"""
evaluate.py
-----------
WHAT THIS MODULE DOES:
  Loads the trained DistilBERT model and computes comprehensive evaluation
  metrics on the held-out test set.

METRICS EXPLAINED:

  1. ACCURACY:
     (TP + TN) / (TP + TN + FP + FN)
     "What fraction of ALL predictions were correct?"
     LIMITATION: misleading on imbalanced classes (if 90% are legit,
     a model that always predicts legit has 90% accuracy but is useless).

  2. PRECISION (per class):
     TP / (TP + FP)
     "Of everything we predicted as phishing, how many actually were?"
     High precision = few false alarms.

  3. RECALL (Sensitivity):
     TP / (TP + FN)
     "Of all real phishing messages, how many did we catch?"
     High recall = few missed phishing.

  4. F1 SCORE:
     2 × (Precision × Recall) / (Precision + Recall)
     Harmonic mean — the best single metric for imbalanced classes.
     Penalizes models that sacrifice one metric for the other.

  5. CONFUSION MATRIX:
     Rows = Actual class, Columns = Predicted class
     ┌─────────────┬──────────────┬──────────────┐
     │             │ Pred: Legit  │ Pred: Phish  │
     ├─────────────┼──────────────┼──────────────┤
     │ Act: Legit  │ TN           │ FP (false alarm) │
     │ Act: Phish  │ FN (missed!) │ TP (caught!) │
     └─────────────┴──────────────┴──────────────┘
     For phishing detection, FN (missed phishing) is more dangerous
     than FP (false alarm) — so we prefer high recall over high precision.

  6. ROC-AUC:
     Area Under the Receiver Operating Characteristic curve.
     Measures model discrimination ability independent of threshold.
     1.0 = perfect, 0.5 = random guessing.
"""

import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess_for_distilbert
from train import PhishingDataset, evaluate, DEVICE, CONFIG


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray) -> dict:
    """
    Compute all required evaluation metrics.

    Args:
        y_true:  Ground truth labels (0/1)
        y_pred:  Predicted labels (0/1)
        y_proba: Predicted probabilities for class 1 (phishing)

    Returns:
        dict of all metrics
    """
    metrics = {
        "accuracy":          accuracy_score(y_true, y_pred),
        "precision_phish":   precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_phish":      recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_phish":          f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_legit":   precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_legit":      recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_legit":          f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_weighted":       f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc":           roc_auc_score(y_true, y_proba),
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_dir: str):
    """
    Plot and save a styled confusion matrix.

    COLOR CODING:
      - Diagonal (correct predictions): darker = better
      - Off-diagonal (errors): we want these near 0
      - Bottom-left cell (FN = missed phishing) is the most dangerous error
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Normalize for percentage display
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    sns.heatmap(
        cm,
        annot    = True,
        fmt      = "d",
        cmap     = "Blues",
        xticklabels = ["Legitimate (0)", "Phishing (1)"],
        yticklabels = ["Legitimate (0)", "Phishing (1)"],
        ax       = ax,
        linewidths  = 0.5,
        cbar_kws = {"label": "Count"},
    )

    # Overlay percentages
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.7, f"({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[i,j] > cm.max()*0.5 else "gray")

    ax.set_title("Confusion Matrix\n(DistilBERT Phishing Classifier)", fontsize=13, pad=12)
    ax.set_ylabel("Actual Label",    fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)

    # Annotate dangerous error cell (FN)
    ax.add_patch(plt.Rectangle((0, 1), 1, 1, fill=False, edgecolor="red",
                                lw=2.5, label="Dangerous: missed phishing"))

    legend_patch = mpatches.Patch(edgecolor="red", facecolor="none", label="Dangerous FN")
    ax.legend(handles=[legend_patch], loc="upper right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {path}")


def plot_roc_curve(y_true, y_proba, save_dir: str):
    """Plot ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc          = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Random (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#1f77b4")

    ax.set_title("ROC Curve — Phishing Detection", fontsize=13)
    ax.set_xlabel("False Positive Rate\n(Fraction of legitimate emails flagged)", fontsize=11)
    ax.set_ylabel("True Positive Rate\n(Fraction of phishing emails caught)", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.01])

    plt.tight_layout()
    path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"ROC curve saved → {path}")


def plot_confidence_distribution(y_true, y_proba, save_dir: str):
    """
    Histogram of model confidence scores per class.
    A well-calibrated model should have high-confidence predictions cluster
    near 0.0 (legit) and 1.0 (phishing) with few uncertain 0.4-0.6 scores.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    phish_probs = y_proba[y_true == 1]
    legit_probs = y_proba[y_true == 0]

    ax.hist(legit_probs, bins=20, alpha=0.7, color="#2196F3",
            label="Legitimate messages", density=True)
    ax.hist(phish_probs, bins=20, alpha=0.7, color="#F44336",
            label="Phishing messages",   density=True)

    ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Decision threshold (0.5)")

    ax.set_title("Model Confidence Distribution\n(P(phishing) per message)", fontsize=13)
    ax.set_xlabel("Predicted probability of phishing", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "confidence_dist.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confidence distribution saved → {path}")


def run_evaluation(model_dir: str, data_path: str, output_dir: str = None):
    """
    Full evaluation pipeline.

    1. Load trained model
    2. Load test data (same split as training)
    3. Get predictions
    4. Compute + print all metrics
    5. Save all plots
    """
    output_dir = output_dir or model_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = os.path.join(model_dir, "best_model")
    print(f"Loading model from {model_path}...")

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model     = DistilBertForSequenceClassification.from_pretrained(model_path).to(DEVICE)

    # ── Load + split data ─────────────────────────────────────────────────────
    df = pd.read_csv(data_path)
    df["text"] = df["text"].astype(str).apply(preprocess_for_distilbert)
    df = df.dropna()

    _, X_val, _, y_val = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=CONFIG["test_size"], random_state=CONFIG["seed"],
        stratify=df["label"],
    )

    # ── If val preds already saved by train.py, load them ────────────────────
    preds_path = os.path.join(model_dir, "val_preds.npy")
    if os.path.exists(preds_path):
        print("Loading saved validation predictions...")
        y_pred  = np.load(os.path.join(model_dir, "val_preds.npy"))
        y_true  = np.load(os.path.join(model_dir, "val_labels.npy"))
        y_proba = np.load(os.path.join(model_dir, "val_probs.npy"))
    else:
        # Re-run inference
        val_dataset = PhishingDataset(X_val, y_val, tokenizer, CONFIG["max_length"])
        val_loader  = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
        _, _, y_pred, y_true, y_proba = evaluate(model, val_loader, DEVICE)

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = compute_metrics(y_true, y_pred, y_proba)

    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS — DISTILBERT PHISHING DETECTOR")
    print("=" * 55)
    print(f"  Accuracy         : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  ROC AUC          : {metrics['roc_auc']:.4f}")
    print(f"")
    print(f"  PHISHING CLASS (label=1):")
    print(f"    Precision       : {metrics['precision_phish']:.4f}")
    print(f"    Recall          : {metrics['recall_phish']:.4f}")
    print(f"    F1 Score        : {metrics['f1_phish']:.4f}")
    print(f"")
    print(f"  LEGITIMATE CLASS (label=0):")
    print(f"    Precision       : {metrics['precision_legit']:.4f}")
    print(f"    Recall          : {metrics['recall_legit']:.4f}")
    print(f"    F1 Score        : {metrics['f1_legit']:.4f}")
    print(f"")
    print(f"  Weighted F1      : {metrics['f1_weighted']:.4f}")
    print("=" * 55)

    print("\nFull Classification Report:")
    print(classification_report(y_true, y_pred,
                                 target_names=["Legitimate", "Phishing"]))

    # ── Save metrics ──────────────────────────────────────────────────────────
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    # ── Generate plots ────────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred, output_dir)
    plot_roc_curve(y_true, y_proba, output_dir)
    plot_confidence_distribution(y_true, y_proba, output_dir)

    return metrics


# ── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE      = os.path.dirname(os.path.dirname(__file__))
    MODEL_DIR = os.path.join(BASE, "models")
    DATA_PATH = os.path.join(BASE, "data", "dataset.csv")

    run_evaluation(MODEL_DIR, DATA_PATH)