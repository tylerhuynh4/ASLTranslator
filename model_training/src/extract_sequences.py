"""
Extract fixed-length landmark sequences for temporal models.
Outputs X (N, T, F), y, lengths, paths, texts.
"""

from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
# Project-level model_training directory (one level above this src file)
MODEL_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH = MODEL_ROOT / "hand_landmarker.task"
POSE_MODEL_PATH = MODEL_ROOT / "pose_landmarker_full.task"
DEFAULT_SPLIT = WORKSPACE_ROOT / "ASL_Citizen" / "splits" / "train.csv"
# Data and output paths are under the model_training root, not the src dir
DEFAULT_NEG_DIR = MODEL_ROOT / "data" / "negatives" / "clips"
DEFAULT_OUT = MODEL_ROOT / "data" / "asl_sequences.npz"
DEFAULT_SUBSET = WORKSPACE_ROOT / "ASL_Citizen" / "subset_glosses.txt"


def load_subset_signs(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def extract_frame_landmarks(
    hand_landmarker,
    pose_landmarker,
    frame: np.ndarray,
    two_hands: bool,
    hand_presence: bool,
) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    # Hand landmarks
    hand_result = hand_landmarker.detect(mp_image)
    # Pose landmarks
    pose_result = pose_landmarker.detect(mp_image)

    # --- Extract reference point (nose) for position normalization ---
    reference = np.array([0.5, 0.5, 0.0], dtype=np.float32)  # default center
    if pose_result.pose_landmarks:
        nose = pose_result.pose_landmarks[0][0]  # index 0 is nose
        reference = np.array([nose.x, nose.y, nose.z], dtype=np.float32)

    # --- Hand features (always fixed length, normalized relative to nose) ---
    if not two_hands:
        # One hand: 21*3
        if hand_result.hand_landmarks:
            landmarks = hand_result.hand_landmarks[0]
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
            coords = coords - reference  # Position-invariant normalization
            hand_feat = coords.reshape(-1)
        else:
            hand_feat = np.zeros(21*3, dtype=np.float32)
    else:
        # Two hands: 2*21*3
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
        left_coords = (
            np.array([[lm.x, lm.y, lm.z] for lm in left], dtype=np.float32) - reference if left else zero
        )
        right_coords = (
            np.array([[lm.x, lm.y, lm.z] for lm in right], dtype=np.float32) - reference if right else zero
        )
        hand_feat = np.concatenate([left_coords.reshape(-1), right_coords.reshape(-1)], axis=0)
        if hand_presence:
            hand_feat = np.concatenate(
                [hand_feat, np.array([1.0 if left else 0.0, 1.0 if right else 0.0], dtype=np.float32)],
                axis=0,
            )

    # --- No pose features (elbows removed as noise) ---
    # Return only hand features normalized relative to nose
    return hand_feat



def extract_clip_sequence(
    hand_landmarker,
    pose_landmarker,
    video_path: Path,
    frame_stride: int,
    seq_len: int,
    two_hands: bool,
    hand_presence: bool,
) -> tuple[np.ndarray, int] | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frames = []
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % frame_stride != 0:
            frame_index += 1
            continue

        feat = extract_frame_landmarks(hand_landmarker, pose_landmarker, frame, two_hands, hand_presence)
        if feat is not None:
            frames.append(feat)

        frame_index += 1

    cap.release()

    if not frames:
        return None

    data = np.stack(frames, axis=0)
    length = min(seq_len, data.shape[0])

    if data.shape[0] > seq_len:
        idx = np.linspace(0, data.shape[0] - 1, seq_len).astype(int)
        data = data[idx]
        length = seq_len
    else:
        pad = np.zeros((seq_len - data.shape[0], data.shape[1]), dtype=np.float32)
        data = np.concatenate([data, pad], axis=0)

    return data, length


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract landmark sequences from ASL_Citizen CSV")
    parser.add_argument("--split", default=str(DEFAULT_SPLIT), help="CSV split file (train.csv, val.csv, test.csv)")
    parser.add_argument("--neg-dir", default=str(DEFAULT_NEG_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--subset-file", default=str(DEFAULT_SUBSET), help="TXT file of glosses to include")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--frame-stride", type=int, default=2)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"Missing model file: {MODEL_PATH}")
        return 1

    WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = WORKSPACE_ROOT / args.split
    if not split_path.exists():
        print(f"Missing split file: {split_path}")
        return 1

    subset_path = Path(args.subset_file)
    if not subset_path.is_absolute():
        subset_path = WORKSPACE_ROOT / args.subset_file
    if not subset_path.exists():
        print(f"Missing subset file: {subset_path}")
        return 1
    subset_signs = load_subset_signs(subset_path)
    if not subset_signs:
        print(f"Subset file is empty: {subset_path}")
        return 1
    subset_index = {name: idx for idx, name in enumerate(subset_signs)}

    # Read CSV and filter items
    items = []
    with split_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            gloss = row["Gloss"].strip()
            if gloss in subset_index:
                items.append({
                    "video_file": row["Video file"].strip(),
                    "gloss": gloss,
                    "label": subset_index[gloss],
                    "participant": row["Participant ID"].strip(),
                })

    neg_dir = Path(args.neg_dir)
    if not neg_dir.is_absolute():
        neg_dir = SCRIPT_DIR.joinpath(args.neg_dir).resolve()
    neg_paths = sorted(neg_dir.glob("*.mp4"))

    BaseOptions = python.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    use_two_hands = True
    use_hand_presence = True


    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2 if use_two_hands else 1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = HandLandmarker.create_from_options(options)
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    sequences = []
    lengths = []
    labels = []
    paths = []
    texts = []

    total_items = len(items)
    total_negs = len(neg_paths)
    kept = 0
    print(f"Starting extraction: {total_items} filtered clips + {total_negs} negatives")

    if tqdm is None:
        print("Tip: Install tqdm for progress bars: python -m pip install tqdm")

    manifest_iter = tqdm(items, desc="ASL_Citizen clips", unit="clip") if tqdm else items
    for item in manifest_iter:
        # Video path is relative to ASL_Citizen/videos/
        video_path = WORKSPACE_ROOT / "ASL_Citizen" / "videos" / item["video_file"]
        result = extract_clip_sequence(
            landmarker,
            pose_landmarker,
            video_path,
            args.frame_stride,
            args.seq_len,
            use_two_hands,
            use_hand_presence,
        )
        if result is None:
            continue
        seq, length = result
        sequences.append(seq)
        lengths.append(length)
        labels.append(int(item["label"]))
        paths.append(str(video_path))
        texts.append(item["gloss"])
        kept += 1

    nonsign_label = len(subset_signs)
    nonsign_count = 0

    neg_iter = tqdm(neg_paths, desc="Negative clips", unit="clip") if tqdm else neg_paths
    for clip_path in neg_iter:
        result = extract_clip_sequence(
            landmarker,
            pose_landmarker,
            clip_path,
            args.frame_stride,
            args.seq_len,
            use_two_hands,
            use_hand_presence,
        )
        if result is None:
            continue
        seq, length = result
        sequences.append(seq)
        lengths.append(length)
        labels.append(nonsign_label)
        paths.append(str(clip_path))
        texts.append("Non-sign")
        nonsign_count += 1
        kept += 1


    landmarker.close()
    pose_landmarker.close()

    if not sequences:
        print("No sequences extracted. Check the videos and model.")
        return 1

    X = np.stack(sequences, axis=0)
    y = np.array(labels, dtype=np.int64)
    lengths_arr = np.array(lengths, dtype=np.int64)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        lengths=lengths_arr,
        paths=np.array(paths),
        texts=np.array(texts),
    )

    meta_path = out_path.with_suffix(".json")
    meta = {
        "samples": int(len(y)),
        "classes": int(len(set(y.tolist()))),
        "feature_dim": int(X.shape[2]),
        "seq_len": args.seq_len,
        "frame_stride": args.frame_stride,
        "two_hands": bool(use_two_hands),
        "hand_presence": bool(use_hand_presence),
        "nonsign_label": int(nonsign_label),
        "nonsign_count": int(nonsign_count),
        "labels": subset_signs + ["Non-sign"],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved dataset: {out_path}")
    print(f"Metadata: {meta_path}")
    print(f"Labels: {meta['labels']}")
    print(f"Samples: {meta['samples']} | Classes: {meta['classes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())