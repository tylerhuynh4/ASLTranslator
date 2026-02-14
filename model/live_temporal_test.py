"""
Live webcam test for the temporal GRU model.
Press 'q' to quit.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "data" / "msasl_temporal_model.pt"
SCALER_PATH = SCRIPT_DIR / "data" / "msasl_temporal_scaler.npz"
LABELS_PATH = SCRIPT_DIR / "data" / "msasl_temporal_labels.json"
HAND_MODEL_PATH = SCRIPT_DIR / "hand_landmarker.task"


def load_labels(path: Path, num_classes: int) -> list[str]:
    if not path.exists():
        return [str(i) for i in range(num_classes)]
    with path.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if len(labels) < num_classes:
        labels = labels + [str(i) for i in range(len(labels), num_classes)]
    return labels[:num_classes]


def detect_hands(landmarker, frame: np.ndarray):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return landmarker.detect(mp_image)


def features_from_result(result, two_hands: bool, hand_presence: bool) -> np.ndarray | None:
    if not result.hand_landmarks:
        return None
    if not two_hands:
        landmarks = result.hand_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        return coords.reshape(-1)

    left = None
    right = None
    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name
        if label == "Left":
            left = hand_landmarks
        elif label == "Right":
            right = hand_landmarks

    if left is None and right is None:
        return None

    zero = np.zeros((21, 3), dtype=np.float32)
    left_coords = (
        np.array([[lm.x, lm.y, lm.z] for lm in left], dtype=np.float32) if left else zero
    )
    right_coords = (
        np.array([[lm.x, lm.y, lm.z] for lm in right], dtype=np.float32) if right else zero
    )
    feat = np.concatenate([left_coords.reshape(-1), right_coords.reshape(-1)], axis=0)
    if hand_presence:
        feat = np.concatenate(
            [feat, np.array([1.0 if left else 0.0, 1.0 if right else 0.0], dtype=np.float32)],
            axis=0,
        )
    return feat


def draw_hand_landmarks(frame: np.ndarray, result) -> None:
    if not result.hand_landmarks:
        return
    height, width = frame.shape[:2]
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name
        label = "Right" if label == "Left" else "Left"
        confidence = handedness[0].score

        for lm in hand_landmarks:
            x = width - int(lm.x * width) - 1
            y = int(lm.y * height)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        for start, end in connections:
            start_lm = hand_landmarks[start]
            end_lm = hand_landmarks[end]
            start_pos = (width - int(start_lm.x * width) - 1, int(start_lm.y * height))
            end_pos = (width - int(end_lm.x * width) - 1, int(end_lm.y * height))
            cv2.line(frame, start_pos, end_pos, (255, 0, 0), 2)

        wrist = hand_landmarks[0]
        wx = width - int(wrist.x * width) - 1
        wy = int(wrist.y * height)
        cv2.putText(
            frame,
            f"{label} ({confidence:.2f})",
            (wx - 30, wy - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Live test for temporal GRU model")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--scaler", default=str(SCALER_PATH))
    parser.add_argument("--labels", default=str(LABELS_PATH))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=0, help="Override sequence length")
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    try:
        import torch
        from torch import nn
    except Exception:
        print("Missing dependency: torch")
        return 2

    model_path = Path(args.model)
    scaler_path = Path(args.scaler)
    labels_path = Path(args.labels)

    if not model_path.exists():
        print(f"Missing model checkpoint: {model_path}")
        return 1
    if not scaler_path.exists():
        print(f"Missing scaler file: {scaler_path}")
        return 1
    if not HAND_MODEL_PATH.exists():
        print(f"Missing hand model file: {HAND_MODEL_PATH}")
        return 1

    device = "cpu"
    if args.device == "cuda":
        device = "cuda"
    elif args.device == "auto" and torch.cuda.is_available():
        device = "cuda"

    checkpoint = torch.load(model_path, map_location=device)
    input_dim = int(checkpoint["input_dim"])
    hidden = int(checkpoint["hidden"])
    layers = int(checkpoint["layers"])
    bidirectional = bool(checkpoint["bidirectional"])
    num_classes = int(checkpoint["num_classes"])
    seq_len = int(checkpoint.get("seq_len", 32))
    if args.seq_len > 0:
        seq_len = args.seq_len

    labels = load_labels(labels_path, num_classes)

    two_hands = False
    hand_presence = False
    if input_dim == 126:
        two_hands = True
    elif input_dim == 128:
        two_hands = True
        hand_presence = True

    scaler = np.load(scaler_path)
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)

    if mean.shape[0] != input_dim or std.shape[0] != input_dim:
        print("Scaler feature dimension does not match model input dimension.")
        return 1

    class GRUModel(nn.Module):
        def __init__(self, input_dim: int, hidden: int, layers: int, bidirectional: bool, num_classes: int):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.0,
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

    model = GRUModel(input_dim, hidden, layers, bidirectional, num_classes).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    BaseOptions = python.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2 if two_hands else 1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Failed to open webcam.")
        landmarker.close()
        return 1

    frame_buffer: deque[np.ndarray] = deque(maxlen=seq_len)
    prob_buffer: deque[np.ndarray] = deque(maxlen=max(1, args.smooth))
    valid_count = 0

    print("Live Temporal Test")
    print("Press 'q' to quit")

    while True:
        success, frame = cap.read()
        if not success:
            break

        display_frame = cv2.flip(frame, 1)
        result = detect_hands(landmarker, frame)
        draw_hand_landmarks(display_frame, result)
        feat = features_from_result(result, two_hands, hand_presence)
        if feat is None:
            frame_buffer.append(np.zeros(input_dim, dtype=np.float32))
        else:
            frame_buffer.append(feat)
            valid_count = min(seq_len, valid_count + 1)

        if len(frame_buffer) < seq_len:
            pad = [np.zeros(input_dim, dtype=np.float32)] * (seq_len - len(frame_buffer))
            seq = np.stack(pad + list(frame_buffer), axis=0)
        else:
            seq = np.stack(frame_buffer, axis=0)

        label_text = "No hand"
        topk_text = ""

        if valid_count >= max(1, args.min_frames):
            seq = (seq - mean[None, :]) / std[None, :]
            x = torch.from_numpy(seq[None, :, :]).float().to(device)
            lengths = torch.tensor([min(valid_count, seq_len)], device=device)
            with torch.no_grad():
                logits = model(x, lengths)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            if args.smooth > 1:
                prob_buffer.append(probs)
                probs = np.mean(np.stack(prob_buffer, axis=0), axis=0)

            topk = min(args.topk, num_classes)
            top_idx = np.argsort(-probs)[:topk]
            top_probs = probs[top_idx]

            best_idx = int(top_idx[0])
            best_prob = float(top_probs[0])
            best_label = labels[best_idx] if best_idx < len(labels) else str(best_idx)

            if best_prob < args.threshold:
                if "Non-sign" in labels:
                    best_label = "Non-sign"
                else:
                    best_label = "Unknown"

            label_text = f"{best_label} ({best_prob:.2f})"
            topk_text = " | ".join(
                [
                    f"{labels[idx] if idx < len(labels) else idx}:{prob:.2f}"
                    for idx, prob in zip(top_idx, top_probs)
                ]
            )

        cv2.putText(
            display_frame,
            label_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        if topk_text:
            cv2.putText(
                display_frame,
                topk_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Live Temporal Test", display_frame)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
