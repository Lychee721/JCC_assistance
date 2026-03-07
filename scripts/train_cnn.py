from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.vision.classic_cnn import ClassicItemCNN
from app.vision.dataset import build_imagefolder


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def save_training_plots(history: list[dict[str, float | int | str]], artifact_dir: Path) -> None:
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_acc = [entry["train_acc"] for entry in history]
    val_acc = [entry["val_acc"] for entry in history]
    generalization_gap = [round(t - v, 4) for t, v in zip(train_acc, val_acc)]

    plt.style.use("ggplot")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="train_loss")
    plt.plot(epochs, val_loss, marker="o", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Classic CNN Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifact_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, marker="o", label="train_acc")
    plt.plot(epochs, val_acc, marker="o", label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Classic CNN Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifact_dir / "accuracy_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, generalization_gap, marker="o", color="#d95f02")
    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Train Acc - Val Acc")
    plt.title("Generalization Gap")
    plt.tight_layout()
    plt.savefig(artifact_dir / "generalization_gap.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the classic CNN item classifier.")
    parser.add_argument("--data-root", default="data/vision/datasets/classifier_slots")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--artifact-dir", default="data/vision/artifacts/classic_cnn")
    args = parser.parse_args()

    train_set = build_imagefolder(Path(args.data_root) / "train", train=True)
    val_set = build_imagefolder(Path(args.data_root) / "val", train=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassicItemCNN(num_classes=len(train_set.classes), dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = artifact_dir / "best.pt"
    history_path = artifact_dir / "history.json"
    classes_path = artifact_dir / "classes.json"
    classes_path.write_text(json.dumps(train_set.classes, ensure_ascii=False, indent=2), encoding="utf-8")

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "device": device.type,
            }
        )
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        save_training_plots(history, artifact_dir)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)

        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} device={device.type}"
        )

    print(f"best weights: {best_weights_path}")
    print(f"best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
