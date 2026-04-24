"""
Train a GRU-based multiclass model on landmark sequences.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = MODEL_ROOT.parent
DEPLOY_MODEL_ROOT = WORKSPACE_ROOT / "model"
DATASET_ROOT = MODEL_ROOT / "data" / "ASL_Citizen"

# Default run config so training can be launched with only:
# python model_training/src/train_model.py
# Change the ends of these paths to point to your desired files.
DEFAULT_TRAIN_DATA = MODEL_ROOT / "data" / "demo2H_sequences_train.npz"
DEFAULT_VAL_DATA = MODEL_ROOT / "data" / "demo2H_sequences_val.npz"
DEFAULT_TEST_DATA = MODEL_ROOT / "data" / "demo2H_sequences_test.npz"
DEFAULT_CLASSES = DATASET_ROOT / "subset_glosses.txt"
DEFAULT_MODEL = DEPLOY_MODEL_ROOT / "demo_model_2hand.pt"
DEFAULT_SCALER = DEPLOY_MODEL_ROOT / "demo_scaler_2hand.npz"
DEFAULT_LABELS = DEPLOY_MODEL_ROOT / "demo_labels_2hand.json"

# Default training hyperparameters for short-command runs.
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_HIDDEN = 256
DEFAULT_LAYERS = 2
DEFAULT_DROPOUT = 0.2
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 5
DEFAULT_MIN_DELTA = 1e-3
DEFAULT_SEED = 42
DEFAULT_GRAD_CLIP = 1.0


def resolve_path(raw_path: str, *, base_dir: Path = WORKSPACE_ROOT) -> Path:
    path_obj = Path(raw_path)
    if path_obj.is_absolute():
        return path_obj
    return (base_dir / path_obj).resolve()


def load_dataset(data_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(data_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    lengths = data["lengths"].astype(np.int64)
    return X, y, lengths


def load_labels(classes_path: Path, non_sign_label: int) -> list[str]:
    if not classes_path.exists():
        return [str(i) for i in range(non_sign_label + 1)]

    if classes_path.suffix.lower() == ".txt":
        with classes_path.open("r", encoding="utf-8") as handle:
            labels = [line.strip() for line in handle if line.strip()]
    else:
        with classes_path.open("r", encoding="utf-8") as handle:
            labels = json.load(handle)

    if len(labels) <= non_sign_label:
        labels = labels + ["Non-sign"]
    else:
        labels[non_sign_label] = "Non-sign"

    return labels


def compute_mean_std(X: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(X.shape[2], dtype=np.float64)
    total_sq = np.zeros(X.shape[2], dtype=np.float64)
    count = 0
    for i in range(X.shape[0]):
        length = int(lengths[i])
        if length <= 0:
            continue
        chunk = X[i, :length]
        total += chunk.sum(axis=0)
        total_sq += (chunk ** 2).sum(axis=0)
        count += length
    mean = total / max(1, count)
    var = total_sq / max(1, count) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean[None, None, :]) / std[None, None, :]


def seed_everything(seed: int) -> None:
    try:
        import torch
    except Exception:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def confusion_matrix_from_preds(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_confusion(cm: np.ndarray) -> tuple[float, float, float]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp + 1e-9)
    recall = np.divide(tp, tp + fn + 1e-9)
    f1 = np.divide(2 * precision * recall, precision + recall + 1e-9)
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))
    return macro_precision, macro_recall, macro_f1


def main() -> int:
    parser = argparse.ArgumentParser(description="Train temporal GRU model")
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA))
    parser.add_argument("--val-data", default=str(DEFAULT_VAL_DATA))
    parser.add_argument("--test-data", default=str(DEFAULT_TEST_DATA))
    parser.add_argument("--classes", default=str(DEFAULT_CLASSES))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--scaler", default=str(DEFAULT_SCALER))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    args = parser.parse_args()

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except Exception:
        print("Missing dependency: torch", file=sys.stderr)
        print("Install with: python -m pip install torch", file=sys.stderr)
        return 2

    seed_everything(args.seed)

    train_data_path = resolve_path(args.train_data)
    val_data_path = resolve_path(args.val_data)
    test_data_path = resolve_path(args.test_data)

    if not train_data_path.exists():
        print(f"Missing train dataset: {train_data_path}")
        return 1
    if not val_data_path.exists():
        print(f"Missing val dataset: {val_data_path}")
        return 1
    if not test_data_path.exists():
        print(f"Missing test dataset: {test_data_path}")
        return 1

    X_train, y_train, len_train = load_dataset(train_data_path)
    X_val, y_val, len_val = load_dataset(val_data_path)
    X_test, y_test, len_test = load_dataset(test_data_path)

    non_sign_label = int(max(y_train.max(), y_val.max(), y_test.max()))
    classes_path = resolve_path(args.classes)
    label_names = load_labels(classes_path, non_sign_label)

    mean, std = compute_mean_std(X_train, len_train)
    X_train = apply_norm(X_train, mean, std)
    X_val = apply_norm(X_val, mean, std)
    X_test = apply_norm(X_test, mean, std)

    input_dim = int(X_train.shape[2])
    num_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
    classes, counts = np.unique(y_train, return_counts=True)
    print("Class sample counts:")
    for label, count in zip(classes, counts):
        label_text = label_names[label] if label < len(label_names) else str(label)
        print(f"  Label {label_text} ({label}): {count} samples")
    class_weights = np.ones(num_classes, dtype=np.float32)
    for label, count in zip(classes, counts):
        class_weights[int(label)] = 1.0 / float(count)
    class_weights = class_weights / class_weights.sum() * num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TemporalDataset(Dataset):
        def __init__(self, X_arr, y_arr, len_arr):
            self.X = X_arr
            self.y = y_arr
            self.lengths = len_arr

        def __len__(self) -> int:
            return self.X.shape[0]

        def __getitem__(self, idx: int):
            return self.X[idx], self.y[idx], self.lengths[idx]

    def collate(batch):
        xs, ys, lens = zip(*batch)
        lens = np.array(lens, dtype=np.int64)
        order = np.argsort(-lens)
        xs = np.stack(xs, axis=0)[order]
        ys = np.array(ys, dtype=np.int64)[order]
        lens = lens[order]
        return (
            torch.from_numpy(xs).float(),
            torch.from_numpy(ys).long(),
            torch.from_numpy(lens).long(),
        )

    class GRUModel(nn.Module):
        def __init__(self, input_dim: int, hidden: int, layers: int, bidirectional: bool, num_classes: int):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=args.dropout if layers > 1 else 0.0,
            )
            out_dim = hidden * (2 if bidirectional else 1)
            self.fc = nn.Linear(out_dim, num_classes)

        def forward(self, x, lengths_tensor):
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_tensor.cpu(), batch_first=True, enforce_sorted=True
            )
            _, h = self.gru(packed)
            if self.gru.bidirectional:
                h = torch.cat([h[-2], h[-1]], dim=1)
            else:
                h = h[-1]
            return self.fc(h)

    train_dataset = TemporalDataset(X_train, y_train, len_train)
    val_dataset = TemporalDataset(X_val, y_val, len_val)
    test_dataset = TemporalDataset(X_test, y_test, len_test)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
    )

    model = GRUModel(input_dim, args.hidden, args.layers, args.bidirectional, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    model_path = resolve_path(args.model)
    scaler_path = resolve_path(args.scaler)
    labels_path = resolve_path(args.labels)

    best_top1 = 0.0
    best_top5 = 0.0
    best_macro_f1 = 0.0
    best_epoch = 0
    bad_epochs = 0

    def evaluate(loader):
        model.eval()
        correct = 0
        total = 0
        top5_correct = 0
        total_eval_loss = 0.0
        all_true = []
        all_pred = []
        with torch.no_grad():
            for batch_x, batch_y, batch_lens in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_lens = batch_lens.to(device)
                logits = model(batch_x, batch_lens)
                loss = criterion(logits, batch_y)
                total_eval_loss += float(loss.item())
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                correct += int((preds == batch_y).sum().item())
                total += int(batch_y.size(0))
                all_true.append(batch_y.cpu().numpy())
                all_pred.append(preds.cpu().numpy())
                if probs.size(1) >= 5:
                    top5 = torch.topk(probs, 5, dim=1).indices
                    top5_correct += int((top5 == batch_y.unsqueeze(1)).any(dim=1).sum().item())

        top1 = correct / total if total else 0.0
        top5 = top5_correct / total if total and num_classes >= 5 else 0.0
        avg_loss = total_eval_loss / max(1, len(loader))
        y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
        y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
        cm = confusion_matrix_from_preds(y_true, y_pred, num_classes) if y_true.size else np.zeros((num_classes, num_classes), dtype=np.int64)
        macro_precision, macro_recall, macro_f1 = metrics_from_confusion(cm)
        return {
            "loss": avg_loss,
            "top1": top1,
            "top5": top5,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y, batch_lens in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_lens = batch_lens.to(device)

            optimizer.zero_grad()
            logits = model(batch_x, batch_lens)
            loss = criterion(logits, batch_y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item())

        val_metrics = evaluate(val_loader)
        avg_loss = total_loss / max(1, len(train_loader))

        print(
            "Epoch {}/{} - train_loss {:.4f} - val_loss {:.4f} - val_top1 {:.4f} - val_top5 {:.4f} - val_macro_f1 {:.4f}".format(
                epoch,
                args.epochs,
                avg_loss,
                val_metrics["loss"],
                val_metrics["top1"],
                val_metrics["top5"],
                val_metrics["macro_f1"],
            )
        )

        scheduler.step(val_metrics["top1"])

        if val_metrics["top1"] > best_top1 + args.min_delta:
            best_top1 = val_metrics["top1"]
            best_top5 = val_metrics["top5"]
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            bad_epochs = 0
            best_path = model_path
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "bidirectional": args.bidirectional,
                    "num_classes": num_classes,
                    "seq_len": int(X_train.shape[1]),
                    "best_top1": float(best_top1),
                    "epoch": int(best_epoch),
                },
                best_path,
            )
        else:
            bad_epochs += 1

        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch} (best top1 {best_top1:.4f} at epoch {best_epoch}).")
            break

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scaler_path, mean=mean, std=std)
    labels_path.write_text(json.dumps(label_names, indent=2), encoding="utf-8")

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(test_loader)

    print(f"Best model saved: {model_path}")
    print(f"Scaler saved: {scaler_path}")
    print(f"Labels file saved: {labels_path}")
    print(
        "Final test metrics - loss {:.4f} - top1 {:.4f} - top5 {:.4f} - macro_f1 {:.4f}".format(
            test_metrics["loss"],
            test_metrics["top1"],
            test_metrics["top5"],
            test_metrics["macro_f1"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())