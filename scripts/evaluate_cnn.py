from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.vision.classic_cnn import ClassicItemCNN
from app.vision.dataset import build_imagefolder


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained classic CNN on the test split.")
    parser.add_argument("--data-root", default="data/vision/datasets/classifier_slots")
    parser.add_argument("--weights", default="data/vision/artifacts/classic_cnn/best.pt")
    parser.add_argument("--classes", default="data/vision/artifacts/classic_cnn/classes.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    class_names = json.loads(Path(args.classes).read_text(encoding="utf-8"))
    test_set = build_imagefolder(Path(args.data_root) / "test", train=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassicItemCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    class_correct = {name: 0 for name in class_names}
    class_total = {name: 0 for name in class_names}

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
                if pred == label:
                    class_correct[class_name] += 1

    print(f"test_acc={correct / max(total, 1):.4f} device={device.type}")
    for class_name in class_names:
        if class_total[class_name]:
            print(f"{class_name}: {class_correct[class_name] / class_total[class_name]:.4f}")


if __name__ == "__main__":
    main()
