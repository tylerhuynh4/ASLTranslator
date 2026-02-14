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
DEFAULT_DATA = SCRIPT_DIR / "data" / "msasl_sequences.npz"
DEFAULT_CLASSES = SCRIPT_DIR.parent / "MS-ASL" / "MSASL_classes.json"
DEFAULT_MODEL = SCRIPT_DIR / "data" / "msasl_temporal_model.pt"
DEFAULT_SCALER = SCRIPT_DIR / "data" / "msasl_temporal_scaler.npz"
DEFAULT_LABELS = SCRIPT_DIR / "data" / "msasl_temporal_labels.json"


def load_labels(classes_path: Path, non_sign_label: int) -> list[str]:
    if not classes_path.exists():
        return [str(i) for i in range(non_sign_label + 1)]

    with classes_path.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)

    if len(labels) <= non_sign_label:
        labels = labels + ["Non-sign"]
    else:
        labels[non_sign_label] = "Non-sign"

    return labels


def stratified_split(y: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        split = int(len(idx) * (1.0 - test_size))
        train_idx.extend(idx[:split])
        test_idx.extend(idx[split:])
    return np.array(train_idx), np.array(test_idx)


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
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--classes", default=str(DEFAULT_CLASSES))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--scaler", default=str(DEFAULT_SCALER))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
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

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Missing dataset: {data_path}")
        return 1

    data = np.load(data_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    lengths = data["lengths"].astype(np.int64)

    non_sign_label = int(y.max())
    label_names = load_labels(Path(args.classes), non_sign_label)

    train_idx, test_idx = stratified_split(y, args.test_size, seed=42)
    X_train, y_train, len_train = X[train_idx], y[train_idx], lengths[train_idx]
    X_test, y_test, len_test = X[test_idx], y[test_idx], lengths[test_idx]

    mean, std = compute_mean_std(X_train, len_train)
    X_train = apply_norm(X_train, mean, std)
    X_test = apply_norm(X_test, mean, std)

    input_dim = int(X_train.shape[2])
    num_classes = int(y.max()) + 1
    classes, counts = np.unique(y_train, return_counts=True)
    class_weights = np.zeros(num_classes, dtype=np.float32)
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
    test_dataset = TemporalDataset(X_test, y_test, len_test)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    model_path = Path(args.model)
    scaler_path = Path(args.scaler)
    labels_path = Path(args.labels)

    best_top1 = 0.0
    best_top5 = 0.0
    best_macro_f1 = 0.0
    best_epoch = 0
    bad_epochs = 0

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

        model.eval()
        correct = 0
        total = 0
        top5_correct = 0
        val_loss = 0.0
        all_true = []
        all_pred = []
        with torch.no_grad():
            for batch_x, batch_y, batch_lens in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_lens = batch_lens.to(device)
                logits = model(batch_x, batch_lens)
                loss = criterion(logits, batch_y)
                val_loss += float(loss.item())
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
        avg_loss = total_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(test_loader))
        y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
        y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
        cm = confusion_matrix_from_preds(y_true, y_pred, num_classes) if y_true.size else np.zeros((num_classes, num_classes), dtype=np.int64)
        macro_precision, macro_recall, macro_f1 = metrics_from_confusion(cm)

        print(
            "Epoch {}/{} - loss {:.4f} - val_loss {:.4f} - top1 {:.4f} - top5 {:.4f} - macro_f1 {:.4f}".format(
                epoch, args.epochs, avg_loss, avg_val_loss, top1, top5, macro_f1
            )
        )

        scheduler.step(top1)

        if top1 > best_top1 + args.min_delta:
            best_top1 = top1
            best_top5 = top5
            best_macro_f1 = macro_f1
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
                    "seq_len": int(X.shape[1]),
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

    np.savez_compressed(scaler_path, mean=mean, std=std)

    labels_path.write_text(json.dumps(label_names, indent=2), encoding="utf-8")

    print(f"Best model saved: {model_path}")
    print(f"Scaler saved: {scaler_path}")
    print(f"Labels saved: {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
