# ============================================================================
# Leaf Classification (Smooth vs Serrated) using PyTorch + ConvNeXt
# Author: Messaoudi-Belkacem
# Date: 2025-11-11
# ============================================================================

# 1. IMPORTS
# ============================================================================
import os, random, math, time
from dataclasses import dataclass
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
import torchvision.models as models
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    balanced_accuracy_score, cohen_kappa_score,
    log_loss, roc_auc_score, classification_report,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay,
    ConfusionMatrixDisplay
)

# Set random seeds for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else 
                      ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"🖥️  Using device: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# 2. DATA CONFIGURATION
# ============================================================================
DATA_DIR = "C:\\Users\\HP\\workspace\\leaf-classifier\\dataset\\feuilles_plantes"

weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
default_transforms = weights.transforms()

# Training transformations
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=default_transforms.mean,
        std=default_transforms.std
    )
])

# Validation/Test transformations (no augmentation)
transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=default_transforms.mean,
        std=default_transforms.std
    )
])

# Load full dataset
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform_train)
num_classes = len(full_dataset.classes)

print(f"\n📁 Dataset Info:")
print(f"   Classes: {full_dataset.classes}")
print(f"   Total images: {len(full_dataset)}")
print(f"   Number of classes: {num_classes}")

# ============================================================================
# 3. STRATIFIED TRAIN/VAL/TEST SPLIT
# ============================================================================
@dataclass
class Splits:
    train: Subset
    val: Subset
    test: Subset

def stratified_split(dataset: datasets.ImageFolder,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     seed: int = 0) -> Splits:
    """Returns train/val/test Subsets with stratified class distribution."""
    rng = np.random.RandomState(seed)
    
    # Group indices by class label
    class_indices = {}
    for idx, (_, label) in enumerate(dataset.imgs):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Split each class proportionally
    for label, indices in class_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        
        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_indices.extend(indices[:n_train].tolist())
        val_indices.extend(indices[n_train:n_train+n_val].tolist())
        test_indices.extend(indices[n_train+n_val:].tolist())
    
    return Splits(
        train=Subset(dataset, train_indices),
        val=Subset(dataset, val_indices),
        test=Subset(dataset, test_indices)
    )

# Create splits
splits = stratified_split(full_dataset, 0.7, 0.15, 0.15, seed=0)

# Create optimized DataLoaders
train_loader = DataLoader(
    splits.train, 
    batch_size=64,
    shuffle=True, 
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

full_dataset.transform = transform_eval
val_loader = DataLoader(
    splits.val, 
    batch_size=256,  # Increased from 64
    shuffle=False, 
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

test_loader = DataLoader(
    splits.test, 
    batch_size=256,
    shuffle=False, 
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

print(f"\n📊 Split sizes:")
print(f"   Train: {len(splits.train)}")
print(f"   Val: {len(splits.val)}")
print(f"   Test: {len(splits.test)}")

# ============================================================================
# 4. MODEL ARCHITECTURE (ConvNeXt)
# ============================================================================
def create_convnext_model(num_classes=2, model_size='tiny'):
    """Create ConvNeXt model with pretrained weights."""
    print(f"\n🏗️  Building ConvNeXt-{model_size} model...")
    
    if model_size == 'tiny':
        model = models.convnext_tiny(weights='IMAGENET1K_V1')
    elif model_size == 'small':
        model = models.convnext_small(weights='IMAGENET1K_V1')
    elif model_size == 'base':
        model = models.convnext_base(weights='IMAGENET1K_V1')
    else:
        model = models.convnext_large(weights='IMAGENET1K_V1')
    
    # Replace classifier head
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    
    return model

model = create_convnext_model(num_classes=num_classes, model_size='tiny')
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")

# ============================================================================
# 5. TRAINING SETUP
# ============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
scaler = GradScaler()

# ============================================================================
# 6. TRAINING & EVALUATION FUNCTIONS
# ============================================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        total_loss += loss.item() * imgs.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100.*correct/total:.1f}%'
        })
    
    return total_loss / len(loader.dataset), 100. * correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evaluate model without gradient computation."""
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_proba = [], [], []
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Validation', bar_format='{l_bar}{bar:30}{r_bar}', leave=False)
    
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_proba.extend(probs[:, 1].cpu().numpy())
        
        total_loss += loss.item() * imgs.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100.*correct/total:.1f}%'
        })
    
    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100. * correct / total
    return avg_loss, np.array(y_true), np.array(y_pred), np.array(y_proba), accuracy

# ============================================================================
# 7. TRAINING LOOP
# ============================================================================
print("\n🚀 Starting training...\n")
best_val_acc = 0.0
epochs = 30

for epoch in range(epochs):
    # Training
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
    
    # Validation
    va_loss, y_t, y_p, y_s, va_acc = evaluate(model, val_loader, criterion)
    
    # Update learning rate
    scheduler.step(va_loss)
    
    # Print epoch summary
    print(f"📈 Epoch {epoch+1:02d}/{epochs} | "
          f"Train: {tr_loss:.4f} ({tr_acc:.2f}%) | "
          f"Val: {va_loss:.4f} ({va_acc:.2f}%) | "
          f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Save best model
    if va_acc > best_val_acc:
        best_val_acc = va_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"   ✅ New best model saved! (Val Acc: {best_val_acc:.2f}%)")
    
    print()

print(f"\n✨ Training complete! Best Val Accuracy: {best_val_acc:.2f}%")

# ============================================================================
# 8. FINAL EVALUATION ON TEST SET
# ============================================================================
print("\n🧪 Evaluating on test set...")

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

te_loss, y_true, y_pred, y_proba, te_acc = evaluate(model, test_loader, criterion)

# Identify positive class
classes = full_dataset.classes
pos_class_name = classes[0]
pos_idx = classes.index(pos_class_name)

# Compute comprehensive metrics
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", pos_label=pos_idx
)
bacc = balanced_accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
ll = log_loss(y_true, np.column_stack([1-y_proba, y_proba]))
auc = roc_auc_score((y_true==pos_idx).astype(int), y_proba)

# Print results
print("\n" + "="*60)
print("🎯 TEST SET PERFORMANCE")
print("="*60)
print(f"Test Loss         : {te_loss:.4f}")
print(f"Accuracy          : {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision         : {prec:.4f}")
print(f"Recall            : {rec:.4f}")
print(f"F1 Score          : {f1:.4f}")
print(f"Balanced Accuracy : {bacc:.4f}")
print(f"Cohen's Kappa     : {kappa:.4f}")
print(f"Log Loss          : {ll:.4f}")
print(f"AUC-ROC           : {auc:.4f}")
print("\n" + "="*60)
print("📋 CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_true, y_pred, target_names=classes))

# ============================================================================
# 9. VISUALIZATIONS
# ============================================================================
print("\n📊 Generating visualizations...")

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=classes).plot(values_format='d', cmap='Blues')
plt.title("Confusion Matrix (Test Set)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC Curve
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_predictions((y_true==pos_idx).astype(int), y_proba)
plt.title("ROC Curve (Test Set)", fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Precision-Recall Curve
plt.figure(figsize=(8, 6))
PrecisionRecallDisplay.from_predictions((y_true==pos_idx).astype(int), y_proba)
plt.title("Precision-Recall Curve (Test Set)", fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ All done! Model saved as 'best_model.pth'")
print(f"📈 Final Test Accuracy: {acc*100:.2f}%")