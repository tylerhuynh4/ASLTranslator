"""
Reusable temporal predictor for real-time ASL inference.

Public API:
    predict(frame) -> (label: str, confidence: float)
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
import sys
from threading import Lock
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEPLOY_MODEL_ROOT = PROJECT_ROOT / "model"
ASSET_ROOT = PROJECT_ROOT / "model_training"
WORKSPACE_ROOT = PROJECT_ROOT

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

DEFAULT_MODEL = DEPLOY_MODEL_ROOT / "asl_citizen_temporal_modelv1.pt"
DEFAULT_SCALER = DEPLOY_MODEL_ROOT / "asl_citizen_temporal_scalerv1.npz"
DEFAULT_LABELS = DEPLOY_MODEL_ROOT / "asl_citizen_temporal_labelsv1.json"
DEFAULT_HAND_MODEL = ASSET_ROOT / "hand_landmarker.task"
DEFAULT_POSE_MODEL = ASSET_ROOT / "pose_landmarker_full.task"


def _resolve_path(raw_path: str | Path, *, base_dir: Path = WORKSPACE_ROOT) -> Path:
    path_obj = Path(raw_path)
    if path_obj.is_absolute():
        return path_obj
    return (base_dir / path_obj).resolve()


def _load_labels(path: Path, num_classes: int) -> list[str]:
    if not path.exists():
        return [str(i) for i in range(num_classes)]

    with path.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)

    if len(labels) < num_classes:
        labels = labels + [str(i) for i in range(len(labels), num_classes)]
    return labels[:num_classes]


def _infer_feature_config(input_dim: int) -> tuple[bool, bool]:
    # 63: one hand | 126: two hands | 128: two hands + presence
    # Keep behavior aligned with live_temporal_test.py.
    two_hands = input_dim in {126, 128, 134, 137}
    hand_presence = input_dim in {128, 134, 137}
    return two_hands, hand_presence


class _GRUModel:
    def __init__(self, nn_module: Any, input_dim: int, hidden: int, layers: int, bidirectional: bool, num_classes: int):
        self._inner = nn_module.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self._fc = nn_module.Linear(out_dim, num_classes)
        self._nn = nn_module

    def __call__(self, x: Any, lengths_tensor: Any) -> Any:
        packed = self._nn.utils.rnn.pack_padded_sequence(
            x,
            lengths_tensor.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        _, h = self._inner(packed)
        if self._inner.bidirectional:
            h = np.concatenate([h[-2].detach().cpu().numpy(), h[-1].detach().cpu().numpy()], axis=1)
            h = self._nn.tensor(h, device=x.device)
        else:
            h = h[-1]
        return self._fc(h)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Build a small shim so saved checkpoints load with the same key names as training.
        self._inner.load_state_dict(
            {
                key.replace("gru.", "", 1): value
                for key, value in state_dict.items()
                if key.startswith("gru.")
            }
        )
        self._fc.load_state_dict(
            {
                key.replace("fc.", "", 1): value
                for key, value in state_dict.items()
                if key.startswith("fc.")
            }
        )

    def eval(self) -> None:
        self._inner.eval()
        self._fc.eval()

    def to(self, device: str) -> "_GRUModel":
        self._inner.to(device)
        self._fc.to(device)
        return self


class TemporalPredictor:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        scaler_path: str | Path = DEFAULT_SCALER,
        labels_path: str | Path = DEFAULT_LABELS,
        hand_model_path: str | Path = DEFAULT_HAND_MODEL,
        pose_model_path: str | Path = DEFAULT_POSE_MODEL,
        min_frames: int = 8,
        smooth: int = 5,
        threshold: float = 0.2,
        device: str = "auto",
    ):
        self.model_path = _resolve_path(model_path)
        self.scaler_path = _resolve_path(scaler_path)
        self.labels_path = _resolve_path(labels_path)
        self.hand_model_path = _resolve_path(hand_model_path)
        self.pose_model_path = _resolve_path(pose_model_path)

        self.min_frames = max(1, int(min_frames))
        self.smooth = max(1, int(smooth))
        self.threshold = float(threshold)
        self.device_pref = device

        self._initialized = False
        self._init_error: str | None = None

        self._lock = Lock()
        self._frame_buffer: deque[np.ndarray] | None = None
        self._prob_buffer: deque[np.ndarray] | None = None
        self._valid_count = 0

        self._cv2 = None
        self._mp = None
        self._python = None
        self._vision = None
        self._torch = None
        self._nn = None

        self.model = None
        self.hand_landmarker = None
        self.pose_landmarker = None

        self.input_dim = 0
        self.seq_len = 32
        self.labels: list[str] = []
        self.non_sign_label = "Non-sign"
        self.mean = np.array([], dtype=np.float32)
        self.std = np.array([], dtype=np.float32)
        self.device = "cpu"
        self.two_hands = True
        self.hand_presence = True

    def _initialize(self) -> None:
        if self._initialized:
            return

        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import torch
            from torch import nn
        except Exception as exc:
            self._init_error = f"Missing dependency: {exc}"
            return

        if not self.model_path.exists():
            self._init_error = f"Missing model checkpoint: {self.model_path}"
            return
        if not self.scaler_path.exists():
            self._init_error = f"Missing scaler file: {self.scaler_path}"
            return
        if not self.hand_model_path.exists():
            self._init_error = f"Missing hand model file: {self.hand_model_path}"
            return
        if not self.pose_model_path.exists():
            self._init_error = f"Missing pose model file: {self.pose_model_path}"
            return

        self._cv2 = cv2
        self._mp = mp
        self._python = python
        self._vision = vision
        self._torch = torch
        self._nn = nn

        if self.device_pref == "cuda":
            self.device = "cuda"
        elif self.device_pref == "auto" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.input_dim = int(checkpoint["input_dim"])
        hidden = int(checkpoint["hidden"])
        layers = int(checkpoint["layers"])
        bidirectional = bool(checkpoint["bidirectional"])
        num_classes = int(checkpoint["num_classes"])
        self.seq_len = int(checkpoint.get("seq_len", 32))

        self.labels = _load_labels(self.labels_path, num_classes)
        if "Non-sign" in self.labels:
            self.non_sign_label = "Non-sign"
        elif self.labels:
            self.non_sign_label = self.labels[-1]

        scaler = np.load(self.scaler_path)
        self.mean = scaler["mean"].astype(np.float32)
        self.std = scaler["std"].astype(np.float32)
        if self.mean.shape[0] != self.input_dim or self.std.shape[0] != self.input_dim:
            self._init_error = "Scaler feature dimension does not match model input dimension"
            return

        self.two_hands, self.hand_presence = _infer_feature_config(self.input_dim)

        self.model = _GRUModel(nn, self.input_dim, hidden, layers, bidirectional, num_classes).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.hand_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2 if self.two_hands else 1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.pose_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        self._frame_buffer = deque(maxlen=self.seq_len)
        self._prob_buffer = deque(maxlen=self.smooth)
        self._valid_count = 0
        self._initialized = True

    def reset(self) -> None:
        with self._lock:
            if self._frame_buffer is not None:
                self._frame_buffer.clear()
            if self._prob_buffer is not None:
                self._prob_buffer.clear()
            self._valid_count = 0

    def close(self) -> None:
        with self._lock:
            if self.hand_landmarker is not None:
                self.hand_landmarker.close()
            if self.pose_landmarker is not None:
                self.pose_landmarker.close()

    def _features_from_results(self, hand_result: Any, pose_result: Any) -> np.ndarray:
        reference = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        if pose_result.pose_landmarks:
            nose = pose_result.pose_landmarks[0][0]
            reference = np.array([nose.x, nose.y, nose.z], dtype=np.float32)

        if not self.two_hands:
            if hand_result.hand_landmarks:
                landmarks = hand_result.hand_landmarks[0]
                coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
                return (coords - reference).reshape(-1)
            return np.zeros(21 * 3, dtype=np.float32)

        left = None
        right = None
        if hand_result.hand_landmarks:
            for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                label = handedness[0].category_name
                if label == "Left":
                    left = hand_landmarks
                elif label == "Right":
                    right = hand_landmarks

        zero = np.zeros((21, 3), dtype=np.float32)
        left_coords = np.array([[lm.x, lm.y, lm.z] for lm in left], dtype=np.float32) - reference if left else zero
        right_coords = np.array([[lm.x, lm.y, lm.z] for lm in right], dtype=np.float32) - reference if right else zero
        hand_feat = np.concatenate([left_coords.reshape(-1), right_coords.reshape(-1)], axis=0)

        if self.hand_presence:
            hand_feat = np.concatenate(
                [hand_feat, np.array([1.0 if left else 0.0, 1.0 if right else 0.0], dtype=np.float32)],
                axis=0,
            )
        return hand_feat

    def predict(self, frame: np.ndarray) -> tuple[str, float]:
        with self._lock:
            if not self._initialized and self._init_error is None:
                self._initialize()

            if self._init_error is not None:
                return self.non_sign_label, 0.0

            if frame is None or frame.ndim != 3:
                return self.non_sign_label, 0.0

            rgb_frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)

            hand_result = self.hand_landmarker.detect(mp_image)
            pose_result = self.pose_landmarker.detect(mp_image)
            feat = self._features_from_results(hand_result, pose_result)

            if feat.shape[0] != self.input_dim:
                return self.non_sign_label, 0.0

            self._frame_buffer.append(feat)
            self._valid_count = min(self.seq_len, self._valid_count + 1)

            if len(self._frame_buffer) < self.seq_len:
                pad = [np.zeros(self.input_dim, dtype=np.float32)] * (self.seq_len - len(self._frame_buffer))
                seq = np.stack(pad + list(self._frame_buffer), axis=0)
            else:
                seq = np.stack(self._frame_buffer, axis=0)

            if self._valid_count < self.min_frames:
                return self.non_sign_label, 0.0

            seq = (seq - self.mean[None, :]) / self.std[None, :]
            x = self._torch.from_numpy(seq[None, :, :]).float().to(self.device)
            lengths = self._torch.tensor([min(self._valid_count, self.seq_len)], device=self.device)

            with self._torch.no_grad():
                logits = self.model(x, lengths)
                probs = self._torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            if self.smooth > 1:
                self._prob_buffer.append(probs)
                probs = np.mean(np.stack(self._prob_buffer, axis=0), axis=0)

            best_idx = int(np.argmax(probs))
            best_prob = float(probs[best_idx])
            best_label = self.labels[best_idx] if best_idx < len(self.labels) else str(best_idx)

            if best_prob < self.threshold:
                best_label = self.non_sign_label

            return best_label, best_prob


_DEFAULT_PREDICTOR: TemporalPredictor | None = None
_DEFAULT_LOCK = Lock()


def get_default_predictor() -> TemporalPredictor:
    global _DEFAULT_PREDICTOR
    with _DEFAULT_LOCK:
        if _DEFAULT_PREDICTOR is None:
            _DEFAULT_PREDICTOR = TemporalPredictor()
        return _DEFAULT_PREDICTOR


def predict(frame: np.ndarray) -> tuple[str, float]:
    return get_default_predictor().predict(frame)


__all__ = ["TemporalPredictor", "get_default_predictor", "predict"]


def _run_cli_demo() -> int:
    # Keep legacy entrypoint behavior while moving runtime orchestration out of this module.
    try:
        from inference.run_live_demo import main as run_live_demo_main
    except ModuleNotFoundError:
        from run_live_demo import main as run_live_demo_main
    return run_live_demo_main()


if __name__ == "__main__":
    raise SystemExit(_run_cli_demo())
