"""
train.py
--------
WHAT THIS DOES:
  Fine-tunes DistilBERT for binary phishing classification.

KEY DEEP LEARNING CONCEPTS EXPLAINED:

1. WHAT IS DISTILBERT?
   DistilBERT is a "distilled" (compressed) version of BERT.
   - BERT has 12 transformer layers, 110M parameters
   - DistilBERT has 6 layers, 66M parameters — 40% smaller, 60% faster
   - Trained via "knowledge distillation": a small model (student) learns
     to mimic a large model's (teacher) output distributions
   - Retains 97% of BERT's language understanding

2. WHAT IS FINE-TUNING?
   DistilBERT is pre-trained on BookCorpus + Wikipedia to predict
   masked words and sentence relationships. It "understands" English.
   Fine-tuning adds a classification head on top and trains it on OUR
   labeled phishing data — we're teaching it a NEW task using existing
   language knowledge.

   Pre-training: "The cat sat on the ___" → "mat"  (general language)
   Fine-tuning:  "Your account is suspended, verify now" → phishing!  (our task)

3. WORDPIECE TOKENIZATION:
   DistilBERT uses WordPiece tokenization — it splits unknown words
   into subword units:
     "phishing" → ["ph", "##ish", "##ing"]
   This handles misspellings and rare words gracefully.

4. ATTENTION MECHANISM:
   Each token attends to every other token via learned weights.
   [CLS] "your" "account" "is" "suspended" "verify" "now" [SEP]
   The model learns: "verify" + "suspended" together → phishing pattern.

5. [CLS] TOKEN:
   DistilBERT adds a special [CLS] (classification) token at the start.
   After all attention layers, the [CLS] embedding captures the full
   sentence meaning. We feed THIS embedding to our classification head.

6. OPTIMIZER — AdamW:
   Adam with weight decay (L2 regularization).
   - Weight decay prevents overfitting by penalizing large weights
   - Typical LR for fine-tuning: 2e-5 (very small — we're nudging
     pre-trained weights, not training from scratch)

7. LINEAR WARMUP SCHEDULE:
   LR starts at 0, linearly increases to peak (2e-5) over warmup_steps,
   then linearly decays to 0.
   WHY? Unstable to start fine-tuning at full LR — the classification
   head has random weights and high gradients could corrupt pre-trained
   knowledge.
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess_for_distilbert

# ── HYPERPARAMETERS ────────────────────────────────────────────────────────────
# These are well-established defaults for DistilBERT fine-tuning.
# For small datasets (< 1000 samples), keep epochs low to avoid overfitting.

CONFIG = {
    "model_name":     "distilbert-base-uncased",
    "max_length":     128,       # Max tokens per message. 128 covers ~95% of emails.
    "batch_size":     16,        # Larger = faster, but needs more RAM. 16 is safe.
    "epochs":         4,         # 3-5 epochs is the standard for fine-tuning
    "learning_rate":  2e-5,      # Classic fine-tuning LR. Lower = safer.
    "weight_decay":   0.01,      # L2 regularization coefficient
    "warmup_ratio":   0.1,       # 10% of steps for LR warmup
    "test_size":      0.2,       # 20% held out for validation
    "seed":           42,
    "num_labels":     2,         # Binary: 0=legit, 1=phishing
}

# Auto-detect device: use GPU (CUDA/MPS) if available, else CPU
DEVICE = (
    torch.device("cuda")  if torch.cuda.is_available()  else
    torch.device("mps")   if torch.backends.mps.is_available() else
    torch.device("cpu")
)
print(f"Using device: {DEVICE}")


# ── DATASET CLASS ──────────────────────────────────────────────────────────────

class PhishingDataset(Dataset):
    """
    PyTorch Dataset for phishing detection.

    CONCEPT — Why a Dataset class?
      PyTorch's DataLoader expects a Dataset object that implements:
      - __len__()     → number of samples
      - __getitem__() → returns one sample (tokenized + label)

      The DataLoader then batches these automatically, handles shuffling,
      and supports multi-threaded loading.
    """

    def __init__(self, texts: list[str], labels: list[int],
                 tokenizer, max_length: int = 128):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = self.labels[idx]

        # TOKENIZATION:
        # padding="max_length"   → pad short sequences to max_length (needed for batching)
        # truncation=True        → cut sequences longer than max_length
        # return_tensors="pt"    → return PyTorch tensors (not numpy)
        # attention_mask         → 1 for real tokens, 0 for padding (model ignores padding)
        encoding = self.tokenizer(
            text,
            max_length     = self.max_length,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(),       # (max_length,)
            "attention_mask": encoding["attention_mask"].squeeze(),  # (max_length,)
            "labels":         torch.tensor(label, dtype=torch.long),
        }


# ── TRAINING LOOP ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device):
    """Single epoch of training."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # FORWARD PASS:
        # DistilBertForSequenceClassification returns (loss, logits)
        # loss is cross-entropy when labels are provided
        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels,
        )
        loss   = outputs.loss
        logits = outputs.logits

        # BACKWARD PASS + GRADIENT UPDATE:
        optimizer.zero_grad()   # Clear previous gradients
        loss.backward()         # Compute gradients via backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip large gradients
        optimizer.step()        # Update weights
        scheduler.step()        # Update learning rate

        # Track metrics
        total_loss += loss.item()
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate on validation/test set. No gradient computation needed."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels,
        )

        total_loss += outputs.loss.item()
        probs       = torch.softmax(outputs.logits, dim=-1)  # Convert logits → probabilities
        preds       = probs.argmax(dim=-1)

        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # P(phishing)

    return (total_loss / len(loader), correct / total,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


def plot_training_history(history: dict, save_dir: str):
    """Save training/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"],  "b-o", label="Train loss")
    ax1.plot(epochs, history["val_loss"],    "r-o", label="Val loss")
    ax1.set_title("Loss per epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"],   "b-o", label="Train acc")
    ax2.plot(epochs, history["val_acc"],     "r-o", label="Val acc")
    ax2.set_title("Accuracy per epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved → {path}")


def train(data_path: str, model_dir: str):
    """
    Full training pipeline.

    Args:
        data_path: Path to dataset CSV with columns 'text', 'label'
        model_dir: Directory to save trained model + artifacts
    """
    os.makedirs(model_dir, exist_ok=True)
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # ── 1. LOAD DATA ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    df = pd.read_csv(data_path)
    assert "text" in df.columns and "label" in df.columns, \
        "Dataset must have 'text' and 'label' columns"

    df["text"] = df["text"].astype(str).apply(preprocess_for_distilbert)
    df = df.dropna(subset=["text", "label"])

    print(f"  Total samples:  {len(df)}")
    print(f"  Phishing (1):   {df['label'].sum()}")
    print(f"  Legit    (0):   {(df['label'] == 0).sum()}")

    # ── 2. TRAIN/VAL SPLIT ────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size    = CONFIG["test_size"],
        random_state = CONFIG["seed"],
        stratify     = df["label"],   # Maintain class ratio in both splits
    )
    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}")

    # ── 3. TOKENIZER + MODEL ──────────────────────────────────────────────────
    print("\n[2/5] Loading DistilBERT tokenizer + model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(CONFIG["model_name"])

    # DistilBertForSequenceClassification adds a 2-class linear head
    # on top of DistilBERT's pooled [CLS] output
    model = DistilBertForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels = CONFIG["num_labels"],
    ).to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ── 4. DATALOADERS ────────────────────────────────────────────────────────
    print("\n[3/5] Building data loaders...")
    train_dataset = PhishingDataset(X_train, y_train, tokenizer, CONFIG["max_length"])
    val_dataset   = PhishingDataset(X_val,   y_val,   tokenizer, CONFIG["max_length"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"], shuffle=False)

    # ── 5. OPTIMIZER + SCHEDULER ──────────────────────────────────────────────
    total_steps   = len(train_loader) * CONFIG["epochs"]
    warmup_steps  = int(total_steps * CONFIG["warmup_ratio"])

    # AdamW: Adam optimizer + decoupled weight decay (Loshchilov & Hutter, 2019)
    optimizer = AdamW(
        model.parameters(),
        lr           = CONFIG["learning_rate"],
        weight_decay = CONFIG["weight_decay"],
    )

    # Linear schedule: warmup then linear decay
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    print(f"  Total steps:  {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    # ── 6. TRAINING LOOP ──────────────────────────────────────────────────────
    print(f"\n[4/5] Training for {CONFIG['epochs']} epochs on {DEVICE}...")
    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        val_loss, val_acc, preds, labels_arr, probs = evaluate(model, val_loader, DEVICE)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{CONFIG['epochs']}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
              f"({elapsed:.1f}s)")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(os.path.join(model_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(model_dir, "best_model"))
            np.save(os.path.join(model_dir, "val_preds.npy"),  preds)
            np.save(os.path.join(model_dir, "val_labels.npy"), labels_arr)
            np.save(os.path.join(model_dir, "val_probs.npy"),  probs)
            print(f"    ✓ Best model saved (val_acc={best_val_acc:.4f})")

    # ── 7. SAVE ARTIFACTS ─────────────────────────────────────────────────────
    print(f"\n[5/5] Saving artifacts...")

    # Save training config
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Save training history
    with open(os.path.join(model_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    plot_training_history(history, model_dir)

    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {model_dir}/best_model/")
    print(f"{'='*50}")

    return model, tokenizer, history


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(__file__))
    DATA_PATH  = os.path.join(BASE, "data", "dataset.csv")
    MODEL_DIR  = os.path.join(BASE, "models")

    # Download dataset if not present (Kaggle API with synthetic fallback)
    if not os.path.exists(DATA_PATH):
        print("Dataset not found — running downloader...")
        sys.path.insert(0, os.path.join(BASE, "data"))
        import download_dataset as dd
        df = dd.download_dataset()
        df.to_csv(DATA_PATH, index=False)
        print(f"Dataset saved → {DATA_PATH}")

    train(DATA_PATH, MODEL_DIR)