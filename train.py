import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time

# ─────────────────────────────────────────────
#  CONFIGURATION  — edit only this section
# ─────────────────────────────────────────────
DATASET_PATH  = r"C:\Computer Vision Project\Dataset\plantvillage dataset\color"
MODEL_SAVE    = r"C:\Computer Vision Project\model\plant_disease_model.pth"
CLASSES_SAVE  = r"C:\Computer Vision Project\model\class_names.json"
PLOTS_FOLDER  = r"C:\Computer Vision Project\model"

BATCH_SIZE    = 32
NUM_EPOCHS    = 15
LEARNING_RATE = 0.001
IMG_SIZE      = 224
VAL_SPLIT     = 0.2   # 20% for validation
# ─────────────────────────────────────────────

def main():
    # ── Device ──────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  AgroVaidya — Plant Disease Model Trainer")
    print(f"{'='*50}")
    print(f"  Device  : {device}")
    if device.type == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*50}\n")

    # ── Transforms ──────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # ── Dataset ─────────────────────────────
    print("Loading dataset...")
    full_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transform)
    class_names  = full_dataset.classes
    num_classes  = len(class_names)
    print(f"  Classes found : {num_classes}")
    print(f"  Total images  : {len(full_dataset)}\n")

    # Save class names for the Flask API later
    os.makedirs(os.path.dirname(CLASSES_SAVE), exist_ok=True)
    with open(CLASSES_SAVE, "w") as f:
        json.dump(class_names, f)
    print(f"  Class names saved → {CLASSES_SAVE}\n")

    # Train / val split
    val_size   = int(VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply val-only transform to val subset
    val_dataset.dataset = datasets.ImageFolder(DATASET_PATH, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"  Training images   : {train_size}")
    print(f"  Validation images : {val_size}\n")

    # ── Model (Transfer Learning — MobileNetV3) ──
    print("Loading pretrained MobileNetV3-Large...")
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

    # Freeze all base layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head with our own
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.Hardswish(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    model = model.to(device)
    print("  Model ready — only classifier head will be trained first.\n")

    # ── Loss & Optimizer ────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ── Training loop ───────────────────────
    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    best_val_acc = 0.0
    print(f"{'─'*50}")
    print(f"  Starting training for {NUM_EPOCHS} epochs")
    print(f"{'─'*50}\n")

    for epoch in range(NUM_EPOCHS):
        start = time.time()

        # — Train —
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        loop = tqdm(train_loader,
                    desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS} [Train]",
                    leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)
            loop.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # — Validate —
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader,
                                       desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS} [Val]  ",
                                       leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)
                val_loss    += loss.item() * images.size(0)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(f"  Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}% | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":       epoch + 1,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_acc":     val_acc,
                "num_classes": num_classes,
            }, MODEL_SAVE)
            print(f"  ✓ Best model saved (val_acc={val_acc*100:.2f}%)\n")

    print(f"\n{'='*50}")
    print(f"  Training complete!")
    print(f"  Best validation accuracy : {best_val_acc*100:.2f}%")
    print(f"  Model saved → {MODEL_SAVE}")
    print(f"{'='*50}\n")

    # ── Plots ───────────────────────────────
    epochs_range = range(1, NUM_EPOCHS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AgroVaidya — Training Results", fontsize=14, fontweight="bold")

    # Accuracy plot
    axes[0].plot(epochs_range, [a*100 for a in history["train_acc"]],
                 label="Train Accuracy", marker="o", color="#3B6D11")
    axes[0].plot(epochs_range, [a*100 for a in history["val_acc"]],
                 label="Val Accuracy",   marker="o", color="#185FA5")
    axes[0].set_title("Accuracy per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(epochs_range, history["train_loss"],
                 label="Train Loss", marker="o", color="#993C1D")
    axes[1].plot(epochs_range, history["val_loss"],
                 label="Val Loss",   marker="o", color="#854F0B")
    axes[1].set_title("Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_FOLDER, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"  Training curves saved → {plot_path}\n")

    # ── Confusion Matrix (on val set) ───────
    print("Generating confusion matrix...")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Running predictions"):
            images = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names, zero_division=0))

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("AgroVaidya — Confusion Matrix Plant Disease Classification", fontsize=13)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_FOLDER, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.show()
    print(f"  Confusion matrix saved → {cm_path}\n")
    print("All done! You can now run the Flask API.")


if __name__ == "__main__":
    main()