from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.vision.classic_cnn import ClassicItemCNN
from app.vision.dataset import build_imagefolder


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title("Classic CNN Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            color = "white" if value > threshold else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained classic CNN on the test split.")
    parser.add_argument("--data-root", default="data/vision/datasets/classifier_slots")
    parser.add_argument("--weights", default="data/vision/artifacts/classic_cnn/best.pt")
    parser.add_argument("--classes", default="data/vision/artifacts/classic_cnn/classes.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--artifact-dir", default="data/vision/artifacts/classic_cnn")
    args = parser.parse_args()

    class_names = json.loads(Path(args.classes).read_text(encoding="utf-8"))
    test_set = build_imagefolder(Path(args.data_root) / "test", train=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassicItemCNN(num_classes=len(class_names))
    try:
        state_dict = torch.load(args.weights, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    class_correct = {name: 0 for name in class_names}
    class_total = {name: 0 for name in class_names}
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int32)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for pred, label in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                class_name = class_names[label]
                class_total[class_name] += 1
                confusion[label, pred] += 1
                if pred == label:
                    class_correct[class_name] += 1

    test_acc = correct / max(total, 1)
    evaluation = {
        "device": device.type,
        "test_acc": round(test_acc, 4),
        "total_samples": total,
        "per_class_accuracy": {},
        "confusion_matrix": confusion.tolist(),
    }

    print(f"test_acc={test_acc:.4f} device={device.type}")
    for class_name in class_names:
        if class_total[class_name]:
            class_acc = class_correct[class_name] / class_total[class_name]
            evaluation["per_class_accuracy"][class_name] = round(class_acc, 4)
            print(f"{class_name}: {class_acc:.4f}")
        else:
            evaluation["per_class_accuracy"][class_name] = None

    (artifact_dir / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_confusion_matrix(confusion, class_names, artifact_dir / "confusion_matrix.png")
    print(f"evaluation json: {artifact_dir / 'evaluation.json'}")
    print(f"confusion matrix: {artifact_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
