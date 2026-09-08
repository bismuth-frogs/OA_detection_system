"""
Phase 2 — Train the knee X-ray OA classifier.

Binary classification: healthy/mild (KL 0-1) vs at-risk (KL 2-4).
Uses transfer learning on a pretrained ResNet18, auto-detects MPS (Mac GPU),
CUDA, or falls back to CPU.

Expects the same folder structure verify_dataset.py checked:
    data_dir/
        train/  0/ 1/ 2/ 3/ 4/   (or "all/" if you didn't have pre-made splits)
        val/    0/ 1/ 2/ 3/ 4/
        test/   0/ 1/ 2/ 3/ 4/

Usage:
    python train.py --data_dir /path/to/dataset --epochs 15 --out_dir ./checkpoints
"""

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# KL grades 0-1 -> class 0 (healthy/mild), KL grades 2-4 -> class 1 (at-risk)
BINARY_MAP = {"0": 0, "1": 0, "2": 1, "3": 1, "4": 1}
BINARY_CLASS_NAMES = ["healthy_mild", "at_risk"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class BinaryRemappedDataset(torch.utils.data.Dataset):
    """Wraps an ImageFolder dataset and remaps its KL-grade labels (0-4) to binary."""

    def __init__(self, image_folder_dataset):
        self.base = image_folder_dataset
        # ImageFolder assigns integer indices alphabetically to folder names;
        # we need to map those integer indices back to KL grade strings first.
        idx_to_class = {v: k for k, v in self.base.class_to_idx.items()}
        self.remap = {
            idx: BINARY_MAP[idx_to_class[idx]]
            for idx in idx_to_class
            if idx_to_class[idx] in BINARY_MAP
        }

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, orig_label = self.base[i]
        return img, self.remap[orig_label]


def load_datasets(data_dir: Path):
    """Handles both pre-split (train/val/test) and flat class-folder layouts."""
    split_names = {"train", "val", "valid", "validation", "test"}
    subdirs = {d.name.lower(): d for d in data_dir.iterdir() if d.is_dir()}
    has_splits = bool(split_names & subdirs.keys())

    if has_splits:
        train_dir = subdirs.get("train")
        val_dir = subdirs.get("val") or subdirs.get("valid") or subdirs.get("validation")
        test_dir = subdirs.get("test")
        if train_dir is None:
            raise SystemExit("Found split folders but no 'train' directory — check dataset layout.")

        train_base = datasets.ImageFolder(train_dir, transform=build_transforms(train=True))
        train_ds = BinaryRemappedDataset(train_base)

        val_ds = None
        if val_dir is not None:
            val_base = datasets.ImageFolder(val_dir, transform=build_transforms(train=False))
            val_ds = BinaryRemappedDataset(val_base)

        test_ds = None
        if test_dir is not None:
            test_base = datasets.ImageFolder(test_dir, transform=build_transforms(train=False))
            test_ds = BinaryRemappedDataset(test_base)

        return train_ds, val_ds, test_ds

    else:
        # Flat layout: single folder of class subfolders -> do our own 70/15/15 split.
        # We build two parallel ImageFolder instances (train-augmented vs eval-only
        # transforms) over the SAME directory, then use identical random_split indices
        # on each so val/test never get training-time augmentation applied to them.
        base_train_tf = datasets.ImageFolder(data_dir, transform=build_transforms(train=True))
        base_eval_tf = datasets.ImageFolder(data_dir, transform=build_transforms(train=False))
        ds_train_tf = BinaryRemappedDataset(base_train_tf)
        ds_eval_tf = BinaryRemappedDataset(base_eval_tf)

        n = len(ds_train_tf)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        n_test = n - n_train - n_val
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(n, generator=generator).tolist()
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_ds = torch.utils.data.Subset(ds_train_tf, train_idx)
        val_ds = torch.utils.data.Subset(ds_eval_tf, val_idx)
        test_ds = torch.utils.data.Subset(ds_eval_tf, test_idx)
        return train_ds, val_ds, test_ds


def compute_class_weights(dataset):
    """Inverse-frequency class weights, for the loss function, to counter imbalance."""
    counts = [0, 0]
    # dataset may be a BinaryRemappedDataset or a torch Subset wrapping one
    for i in range(len(dataset)):
        _, label = dataset[i]
        counts[label] += 1
    total = sum(counts)
    weights = [total / (2 * c) if c > 0 else 0.0 for c in counts]
    print(f"Class counts (train): healthy_mild={counts[0]}, at_risk={counts[1]}")
    print(f"Class weights applied to loss: {weights}")
    return torch.tensor(weights, dtype=torch.float32)


def build_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze backbone initially -- we'll fine-tune after a warmup
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(BINARY_CLASS_NAMES))
    return model.to(device)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    torch.set_grad_enabled(True)
    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train binary OA risk classifier on knee X-rays.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_head", type=float, default=1e-3, help="LR for classifier head warmup phase")
    parser.add_argument("--lr_finetune", type=float, default=1e-4, help="LR for full fine-tune phase")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Epochs training only the new head")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds = load_datasets(data_dir)
    if val_ds is None:
        raise SystemExit("No validation split found/created — cannot train without one.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(device)

    best_val_acc = 0.0
    best_state = None
    history = []

    # --- Phase A: train classifier head only (backbone frozen) ---
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr_head)
    print(f"\n=== Warmup: training head only for {args.warmup_epochs} epochs ===")
    for epoch in range(args.warmup_epochs):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - start
        print(f"[head {epoch+1}/{args.warmup_epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} ({elapsed:.1f}s)")
        history.append(("head", epoch + 1, train_loss, train_acc, val_loss, val_acc))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # --- Phase B: unfreeze and fine-tune the full network ---
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_finetune)

    remaining_epochs = max(args.epochs - args.warmup_epochs, 0)
    print(f"\n=== Fine-tuning: full network for {remaining_epochs} epochs ===")
    for epoch in range(remaining_epochs):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - start
        print(f"[finetune {epoch+1}/{remaining_epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} ({elapsed:.1f}s)")
        history.append(("finetune", epoch + 1, train_loss, train_acc, val_loss, val_acc))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # Save best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = out_dir / "oa_resnet18_binary.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": BINARY_CLASS_NAMES,
        "best_val_acc": best_val_acc,
    }, ckpt_path)
    print(f"\nBest val_acc: {best_val_acc:.3f}")
    print(f"Saved best checkpoint to: {ckpt_path}")

    # Final test evaluation, if a test split exists
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        print(f"Test set: loss={test_loss:.4f} acc={test_acc:.3f}")
    else:
        print("No test split found — skipping final test evaluation.")

    print("\nNext step: run grad_cam.py against this checkpoint to generate heatmaps.")


if __name__ == "__main__":
    main()
