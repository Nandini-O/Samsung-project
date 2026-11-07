"""
src/model_trainer.py
Phase 5: AI Model Development & Training

Trains, validates, checkpoints, and evaluates MobileNetV2 on the Enhanced Indian Food
43-class dataset using stratified, leakage-free splits and class-weighted CrossEntropyLoss.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

import sys
from pathlib import Path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.vision_dataset import EnhancedFoodDataset, FoodLensDataset, get_vision_transforms

MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
SPLITS_DIR = Path("data/processed/vision_splits")
CACHE_DIR = Path("data/processed/images/enhanced_cache")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_mobilenet_v2(num_classes: int = 43, pretrained: bool = True) -> nn.Module:
    """
    Constructs MobileNetV2 with ImageNet pretrained weights and customized 43-class head.
    """
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return model


def get_class_weights(device: torch.device) -> torch.Tensor:
    """Loads inverse-frequency class weights from models/class_weights.json."""
    weights_path = MODELS_DIR / "class_weights.json"
    with open(weights_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    num_classes = len(data)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for class_name, meta in data.items():
        idx = meta["class_idx"]
        weights[idx] = meta["weight"]
    return weights.to(device)


def accuracy_topk(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> List[float]:
    """Computes top-k accuracy for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    total_samples = 0

    t0 = time.time()
    for batch_idx, (images, targets, _, _) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        batch_size = targets.size(0)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        top1, top5 = accuracy_topk(outputs, targets, topk=(1, 5))
        running_loss += loss.item() * batch_size
        running_top1 += top1 * batch_size
        running_top5 += top5 * batch_size
        total_samples += batch_size

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            curr_loss = running_loss / total_samples
            curr_top1 = running_top1 / total_samples
            curr_top5 = running_top5 / total_samples
            elapsed = time.time() - t0
            print(f"  [Epoch {epoch:02d} | Batch {batch_idx+1:03d}/{len(loader):03d}] "
                  f"Loss: {curr_loss:.4f} | Top-1: {curr_top1:.2f}% | Top-5: {curr_top5:.2f}% | Elapsed: {elapsed:.1f}s")

    epoch_loss = running_loss / total_samples
    epoch_top1 = running_top1 / total_samples
    epoch_top5 = running_top5 / total_samples
    return epoch_loss, epoch_top1, epoch_top5


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str] = None
) -> Dict:
    """Evaluates the model on validation or test set."""
    model.eval()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _, _ in loader:
            images, targets = images.to(device), targets.to(device)
            batch_size = targets.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)

            probs = torch.softmax(outputs, dim=1)
            top1, top5 = accuracy_topk(outputs, targets, topk=(1, 5))

            running_loss += loss.item() * batch_size
            running_top1 += top1 * batch_size
            running_top5 += top5 * batch_size
            total_samples += batch_size

            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    val_loss = running_loss / total_samples
    val_top1 = running_top1 / total_samples
    val_top5 = running_top5 / total_samples

    macro_prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    weighted_prec = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    weighted_rec = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    cm = confusion_matrix(all_targets, all_preds)

    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    ) if class_names else {}

    return {
        "loss": float(val_loss),
        "top1_acc": float(val_top1),
        "top5_acc": float(val_top5),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_prec),
        "weighted_recall": float(weighted_rec),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "predictions": all_preds.tolist(),
        "targets": all_targets.tolist()
    }


def plot_training_dynamics(history: Dict, save_path: Path):
    """Plots training and validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", lw=2)
    axes[0].plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#d62728", lw=2)
    axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Weighted CrossEntropy Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Accuracy
    axes[1].plot(epochs, history["train_top1"], "o-", label="Train Top-1", color="#1f77b4", lw=2)
    axes[1].plot(epochs, history["val_top1"], "s-", label="Val Top-1", color="#2ca02c", lw=2)
    axes[1].plot(epochs, history["val_top5"], "^--", label="Val Top-5", color="#ff7f0e", lw=1.8)
    axes[1].set_title("Training & Validation Accuracy (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    """Plots normalized confusion matrix heatmap for 43 classes."""
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(18, 16))
    cax = ax.matshow(cm_norm, cmap="Blues", interpolation="nearest")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=12, fontweight="bold")
    ax.set_title("Enhanced Indian Food 43-Class Test Confusion Matrix (Normalized)", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix heatmap to {save_path}")


def plot_per_class_f1(report: Dict, class_names: List[str], save_path: Path):
    """Plots per-class F1-scores as a horizontal bar chart."""
    f1s = [report[c]["f1-score"] * 100 for c in class_names if c in report]
    names = [c for c in class_names if c in report]

    # Sort by F1 score
    sorted_pairs = sorted(zip(names, f1s), key=lambda x: x[1])
    sorted_names, sorted_f1s = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 14))
    colors = ["#d62728" if f < 70 else "#2ca02c" if f >= 85 else "#1f77b4" for f in sorted_f1s]
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_f1s, color=colors, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel("Test F1-Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Test Set Per-Class F1-Score (43 Classes)", fontsize=13, fontweight="bold")
    ax.axvline(x=70, color="gray", linestyle="--", alpha=0.7, label="70% Threshold")
    ax.axvline(x=85, color="green", linestyle="--", alpha=0.7, label="85% Threshold")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")

    for i, v in enumerate(sorted_f1s):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved per-class F1 distribution to {save_path}")


def train_and_evaluate(
    stage1_epochs: int = 3,
    stage2_epochs: int = 3,
    batch_size: int = 32,
    num_workers: int = 0
):
    """
    Executes the full training, checkpointing, and evaluation workflow.
    """
    set_seed(42)
    torch.set_num_threads(8)

    device = torch.device("cpu")
    print("=" * 65)
    print("NUTRISENSE AI: PHASE 5 MODEL TRAINING & EVALUATION")
    print(f"Hardware Target: CPU ({torch.get_num_threads()} active threads)")
    print(f"Architecture: MobileNetV2 (Pretrained ImageNet-1K)")
    print(f"Loss Function: Class-Weighted CrossEntropyLoss")
    print("=" * 65)

    # 1. Load Data
    train_tf = get_vision_transforms("train")
    eval_tf = get_vision_transforms("val")

    train_ds = EnhancedFoodDataset(SPLITS_DIR / "train_split.csv", transform=train_tf)
    val_ds = EnhancedFoodDataset(SPLITS_DIR / "val_split.csv", transform=eval_tf)
    test_ds = EnhancedFoodDataset(SPLITS_DIR / "test_split.csv", transform=eval_tf)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = [train_ds.idx_to_class[i] for i in range(len(train_ds.idx_to_class))]
    print(f"Dataset Loaded: {len(train_ds):,} Train | {len(val_ds):,} Val | {len(test_ds):,} Test")

    # 2. Setup Model & Loss
    model = build_mobilenet_v2(num_classes=43, pretrained=True).to(device)
    class_weights = get_class_weights(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_checkpoint_path = MODELS_DIR / "mobilenetv2_food43_best.pth"
    history = {
        "train_loss": [],
        "train_top1": [],
        "train_top5": [],
        "val_loss": [],
        "val_top1": [],
        "val_top5": [],
        "val_macro_f1": []
    }
    best_val_f1 = 0.0
    total_epochs = stage1_epochs + stage2_epochs

    # ==========================================
    # STAGE 1: Train Classifier Head (Feature Extractor Frozen)
    # ==========================================
    print("\n" + "=" * 65)
    print(f"STAGE 1: Feature Extractor Frozen | Training Classifier Head ({stage1_epochs} epochs)")
    print("=" * 65)
    for p in model.features.parameters():
        p.requires_grad = False

    optimizer_head = AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, stage1_epochs + 1):
        print(f"\n--- Epoch {epoch}/{total_epochs} (Stage 1: Head) ---")
        tr_loss, tr_top1, tr_top5 = train_one_epoch(model, train_loader, criterion, optimizer_head, device, epoch)
        val_res = evaluate(model, val_loader, criterion, device, class_names)

        history["train_loss"].append(tr_loss)
        history["train_top1"].append(tr_top1)
        history["train_top5"].append(tr_top5)
        history["val_loss"].append(val_res["loss"])
        history["val_top1"].append(val_res["top1_acc"])
        history["val_top5"].append(val_res["top5_acc"])
        history["val_macro_f1"].append(val_res["macro_f1"])

        print(f"Validation: Loss={val_res['loss']:.4f} | Top-1={val_res['top1_acc']:.2f}% | "
              f"Top-5={val_res['top5_acc']:.2f}% | Macro F1={val_res['macro_f1']*100:.2f}%")

        if val_res["macro_f1"] > best_val_f1:
            best_val_f1 = val_res["macro_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_head.state_dict(),
                "val_metrics": val_res,
                "class_to_idx": train_ds.class_to_idx,
                "architecture": "mobilenet_v2",
                "num_classes": 43
            }, best_checkpoint_path)
            print(f"  --> Checkpoint saved (New Best Val Macro F1: {best_val_f1*100:.2f}%)")

    # ==========================================
    # STAGE 2: Fine-Tuning Top Blocks + Classifier
    # ==========================================
    print("\n" + "=" * 65)
    print(f"STAGE 2: Unfreezing Top Feature Blocks for Fine-Tuning ({stage2_epochs} epochs)")
    print("=" * 65)
    # Unfreeze top inverted residual blocks (features 14 through 18)
    for p in model.features[14:].parameters():
        p.requires_grad = True

    optimizer_ft = AdamW([
        {"params": model.features[14:].parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 3e-4}
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer_ft, T_max=stage2_epochs, eta_min=1e-5)

    for epoch in range(stage1_epochs + 1, total_epochs + 1):
        print(f"\n--- Epoch {epoch}/{total_epochs} (Stage 2: Fine-Tuning) ---")
        tr_loss, tr_top1, tr_top5 = train_one_epoch(model, train_loader, criterion, optimizer_ft, device, epoch)
        scheduler.step()
        val_res = evaluate(model, val_loader, criterion, device, class_names)

        history["train_loss"].append(tr_loss)
        history["train_top1"].append(tr_top1)
        history["train_top5"].append(tr_top5)
        history["val_loss"].append(val_res["loss"])
        history["val_top1"].append(val_res["top1_acc"])
        history["val_top5"].append(val_res["top5_acc"])
        history["val_macro_f1"].append(val_res["macro_f1"])

        print(f"Validation: Loss={val_res['loss']:.4f} | Top-1={val_res['top1_acc']:.2f}% | "
              f"Top-5={val_res['top5_acc']:.2f}% | Macro F1={val_res['macro_f1']*100:.2f}%")

        if val_res["macro_f1"] > best_val_f1:
            best_val_f1 = val_res["macro_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_ft.state_dict(),
                "val_metrics": val_res,
                "class_to_idx": train_ds.class_to_idx,
                "architecture": "mobilenet_v2",
                "num_classes": 43
            }, best_checkpoint_path)
            print(f"  --> Checkpoint saved (New Best Val Macro F1: {best_val_f1*100:.2f}%)")

    # Save History and Curves
    with open(MODELS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_training_dynamics(history, FIGURES_DIR / "12_training_dynamics.png")

    # ==========================================
    # FINAL EVALUATION ON HELD-OUT TEST SET
    # ==========================================
    print("\n" + "=" * 65)
    print("EVALUATING BEST CHECKPOINT ON HELD-OUT TEST SET (2,311 IMAGES)")
    print("=" * 65)
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best checkpoint from Epoch {checkpoint['epoch']} (Best Val Macro F1: {checkpoint['val_metrics']['macro_f1']*100:.2f}%)")

    test_res = evaluate(model, test_loader, criterion, device, class_names)

    print("\n" + "=" * 65)
    print("HELD-OUT TEST SET PERFORMANCE METRICS")
    print("=" * 65)
    print(f"  Test Loss:           {test_res['loss']:.4f}")
    print(f"  Top-1 Accuracy:      {test_res['top1_acc']:.2f}%")
    print(f"  Top-5 Accuracy:      {test_res['top5_acc']:.2f}%")
    print(f"  Macro Precision:     {test_res['macro_precision']*100:.2f}%")
    print(f"  Macro Recall:        {test_res['macro_recall']*100:.2f}%")
    print(f"  Macro F1-Score:      {test_res['macro_f1']*100:.2f}%")
    print(f"  Weighted Precision:  {test_res['weighted_precision']*100:.2f}%")
    print(f"  Weighted Recall:     {test_res['weighted_recall']*100:.2f}%")
    print(f"  Weighted F1-Score:   {test_res['weighted_f1']*100:.2f}%")
    print("=" * 65)

    # Plot Confusion Matrix and Per-Class F1
    cm_arr = np.array(test_res["confusion_matrix"])
    plot_confusion_matrix(cm_arr, class_names, FIGURES_DIR / "13_confusion_matrix.png")
    plot_per_class_f1(test_res["classification_report"], class_names, FIGURES_DIR / "14_per_class_f1_distribution.png")

    # Auxiliary FoodLens Benchmark
    print("\n" + "=" * 65)
    print("EVALUATING ON AUXILIARY FOODLENS TEST SPLIT (CROSS-DATASET GENERALIZATION)")
    print("=" * 65)
    fl_test_ds = FoodLensDataset(SPLITS_DIR / "food_lens_test_split.csv", transform=eval_tf, class_to_idx=train_ds.class_to_idx)
    fl_loader = torch.utils.data.DataLoader(fl_test_ds, batch_size=batch_size, shuffle=False)

    fl_preds = []
    fl_targets = []
    with torch.no_grad():
        for images, targets, _, _ in fl_loader:
            images = images.to(device)
            outputs = model(images)
            fl_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            fl_targets.extend(targets.numpy())

    fl_preds = np.array(fl_preds)
    fl_targets = np.array(fl_targets)
    # Filter to only overlapping classes (target >= 0)
    valid_mask = fl_targets >= 0
    if valid_mask.sum() > 0:
        fl_acc = (fl_preds[valid_mask] == fl_targets[valid_mask]).mean() * 100
        print(f"FoodLens Overlapping Classes Test Accuracy: {fl_acc:.2f}% ({valid_mask.sum()} images)")
    else:
        fl_acc = 0.0

    # Save comprehensive evaluation results
    eval_results = {
        "best_epoch": checkpoint["epoch"],
        "architecture": "MobileNetV2",
        "parameters": sum(p.numel() for p in model.parameters()),
        "test_metrics": {
            "loss": test_res["loss"],
            "top1_accuracy": test_res["top1_acc"],
            "top5_accuracy": test_res["top5_acc"],
            "macro_precision": test_res["macro_precision"],
            "macro_recall": test_res["macro_recall"],
            "macro_f1": test_res["macro_f1"],
            "weighted_precision": test_res["weighted_precision"],
            "weighted_recall": test_res["weighted_recall"],
            "weighted_f1": test_res["weighted_f1"]
        },
        "auxiliary_foodlens_accuracy": fl_acc,
        "per_class_metrics": test_res["classification_report"]
    }

    with open(MODELS_DIR / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nEvaluation results saved to {MODELS_DIR / 'evaluation_results.json'}")
    return eval_results


if __name__ == "__main__":
    train_and_evaluate(stage1_epochs=2, stage2_epochs=2, batch_size=64)
