import os
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.dataset import MRISliceDataset, create_patient_level_splits
from src.model import build_model

def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp + 1e-8)

def plot_training_curves(history, out_path="d:/EDI_PPMI_dataset/data/plots/training_curves.png"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss Curve
    ax1.plot(history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_title("ResNet18 Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy Curve
    ax2.plot(history["train_acc"], label="Train Accuracy", color="#2ca02c", lw=2)
    ax2.plot(history["val_acc"], label="Val Accuracy", color="#d62728", lw=2, linestyle="--")
    ax2.set_title("ResNet18 Classification Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {out_path}")

def plot_confusion_matrix_heatmap(cm, out_path="d:/EDI_PPMI_dataset/data/plots/confusion_matrix.png"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(cax)

    labels = ["Healthy Control", "Parkinson's"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white", fontsize=14, fontweight="bold")

    plt.title("ResNet18 Confusion Matrix (Test Set)", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Diagnosis", fontsize=11)
    plt.ylabel("True Clinical Diagnosis", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {out_path}")

def train_and_evaluate(epochs=15, batch_size=8, lr=3e-4, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load or create splits
    splits_json = "d:/EDI_PPMI_dataset/data/dataset_splits.json"
    if not os.path.exists(splits_json):
        splits = create_patient_level_splits()
    else:
        with open(splits_json, "r") as f:
            splits = json.load(f)

    # Datasets & DataLoaders
    train_ds = MRISliceDataset(splits["train"], splits["labels"], is_train=True)
    val_ds = MRISliceDataset(splits["val"], splits["labels"], is_train=False)
    test_ds = MRISliceDataset(splits["test"], splits["labels"], is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Compute class weights for loss balancing
    train_labels = [splits["labels"][sub] for sub in splits["train"]]
    class_counts = np.bincount(train_labels)
    class_weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum()

    # Model, Loss, Optimizer
    model = build_model(num_classes=2, pretrained=True, device=device)
    
    # Freeze full ResNet18 backbone to act as robust feature extractor
    for name, param in model.backbone.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=1e-2)
    # Apply class weights to penalize missed Parkinson's cases
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_val_metric = -1.0
    best_model_path = "d:/EDI_PPMI_dataset/data/best_resnet18.pth"

    print("\nStarting Transfer Learning ResNet18 Training Loop...")
    print(f"Class Weights: HC={class_weights[0]:.3f}, PD={class_weights[1]:.3f}")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        # TRAIN PHASE
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            outputs = model(x_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_b.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_b.cpu().numpy())

        scheduler.step()

        epoch_train_loss = running_loss / len(train_ds)
        epoch_train_acc = accuracy_score(all_targets, all_preds) * 100.0

        # VAL PHASE (Volume Multi-Triplet Voting Evaluation)
        model.eval()
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for i in range(len(val_ds)):
                vol_t, y_t = val_ds.get_volume_triplets(i)
                vol_t = vol_t.to(device)
                outputs = model(vol_t)
                probs = torch.softmax(outputs, dim=1).mean(dim=0)
                val_probs.append(probs[1].item()) # Probability of Parkinson's
                val_targets.append(y_t.item())

        val_preds = [1 if p >= 0.40 else 0 for p in val_probs]
        epoch_val_loss = 0.0 # Multi-triplet voting metric
        epoch_val_acc = accuracy_score(val_targets, val_preds) * 100.0
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.1f}% (F1: {val_f1:.3f})")

        # Select model based on composite val accuracy and F1 score
        val_metric = epoch_val_acc + (val_f1 * 50.0)
        if val_metric >= best_val_metric:
            best_val_metric = val_metric
            torch.save(model.state_dict(), best_model_path)

    print("=" * 60)
    print("TRAINING COMPLETE! Tuning Decision Threshold for High Clinical Recall (Sensitivity)...")

    # Load Best Model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Find Optimal Decision Threshold for High Clinical Recall
    val_probs = []
    val_targets = []
    with torch.no_grad():
        for i in range(len(val_ds)):
            vol_t, y_t = val_ds.get_volume_triplets(i)
            vol_t = vol_t.to(device)
            outputs = model(vol_t)
            probs = torch.softmax(outputs, dim=1).mean(dim=0)
            val_probs.append(probs[1].item())
            val_targets.append(y_t.item())

    best_thresh = 0.35
    best_score = -1.0
    for th in np.linspace(0.25, 0.50, 26):
        th_preds = [1 if p >= th else 0 for p in val_probs]
        rec = recall_score(val_targets, th_preds, zero_division=0)
        prec = precision_score(val_targets, th_preds, zero_division=0)
        acc = accuracy_score(val_targets, th_preds)
        score = (2.5 * rec) + prec + (0.5 * acc)
        if score >= best_score:
            best_score = score
            best_thresh = th

    print(f"Optimal Decision Threshold (Validation Set): {best_thresh:.3f} (Val Score: {best_score:.3f})")

    # TEST EVALUATION
    test_probs = []
    test_targets = []

    with torch.no_grad():
        for i in range(len(test_ds)):
            vol_t, y_t = test_ds.get_volume_triplets(i)
            vol_t = vol_t.to(device)
            outputs = model(vol_t)
            probs = torch.softmax(outputs, dim=1).mean(dim=0)
            test_probs.append(probs[1].item())
            test_targets.append(y_t.item())

    test_preds = [1 if p >= best_thresh else 0 for p in test_probs]

    acc = accuracy_score(test_targets, test_preds)
    prec = precision_score(test_targets, test_preds, zero_division=0)
    rec = recall_score(test_targets, test_preds, zero_division=0)
    spec = calculate_specificity(test_targets, test_preds)
    f1 = f1_score(test_targets, test_preds, zero_division=0)
    cm = confusion_matrix(test_targets, test_preds, labels=[0, 1])

    metrics_df = pd.DataFrame([{
        "Accuracy": acc,
        "Precision": prec,
        "Recall_Sensitivity": rec,
        "Specificity": spec,
        "F1_Score": f1
    }])

    metrics_csv = "d:/EDI_PPMI_dataset/data/midsem_test_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION METRICS (MIDSEM DEMO)")
    print("=" * 60)
    print(f"  Accuracy:    {acc * 100:.2f}%")
    print(f"  Precision:   {prec * 100:.2f}%")
    print(f"  Sensitivity: {rec * 100:.2f}%")
    print(f"  Specificity: {spec * 100:.2f}%")
    print(f"  F1 Score:    {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("=" * 60)

    # Plot & Save Visuals
    plot_training_curves(history)
    plot_confusion_matrix_heatmap(cm)

    return metrics_df

if __name__ == "__main__":
    train_and_evaluate(epochs=10, batch_size=8, lr=1e-4)
