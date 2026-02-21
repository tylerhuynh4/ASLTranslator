"""
Extract fixed-length landmark sequences for temporal models.
Outputs X (N, T, F), y, lengths, paths, texts.
"""

from __future__ import annotations

import argparse
import json
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
MODEL_PATH = SCRIPT_DIR / "hand_landmarker.task"
POSE_MODEL_PATH = SCRIPT_DIR / "pose_landmarker_full.task"
DEFAULT_MANIFEST = SCRIPT_DIR / "data" / "msasl_all" / "manifest.jsonl"
DEFAULT_NEG_DIR = SCRIPT_DIR / "data" / "negatives" / "clips"
DEFAULT_OUT = SCRIPT_DIR / "data" / "msasl_sequences.npz"
DEFAULT_CLASSES = SCRIPT_DIR.parent / "MS-ASL" / "MSASL_classes.json"
DEFAULT_SUBSET = SCRIPT_DIR / "data" / "subset_signs.json"


def load_manifest(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_label_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_subset_signs(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [str(item).strip() for item in data if str(item).strip()]




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

    # --- Hand features (always fixed length) ---
    if not two_hands:
        # One hand: 21*3
        if hand_result.hand_landmarks:
            landmarks = hand_result.hand_landmarks[0]
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
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
            np.array([[lm.x, lm.y, lm.z] for lm in left], dtype=np.float32) if left else zero
        )
        right_coords = (
            np.array([[lm.x, lm.y, lm.z] for lm in right], dtype=np.float32) if right else zero
        )
        hand_feat = np.concatenate([left_coords.reshape(-1), right_coords.reshape(-1)], axis=0)
        if hand_presence:
            hand_feat = np.concatenate(
                [hand_feat, np.array([1.0 if left else 0.0, 1.0 if right else 0.0], dtype=np.float32)],
                axis=0,
            )

    # --- Pose features (nose, left elbow, right elbow as example, always fixed length) ---
    important_indices = [0, 13, 14]
    pose_coords = []
    if pose_result.pose_landmarks:
        pose_landmarks = pose_result.pose_landmarks[0]
        for idx in important_indices:
            lm = pose_landmarks[idx]
            pose_coords.extend([lm.x, lm.y, lm.z])
    else:
        pose_coords = [0.0] * (len(important_indices) * 3)
    pose_feat = np.array(pose_coords, dtype=np.float32)

    # --- Combine features (always same length) ---
    return np.concatenate([hand_feat, pose_feat], axis=0)



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
    parser = argparse.ArgumentParser(description="Extract landmark sequences")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--neg-dir", default=str(DEFAULT_NEG_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--classes", default=str(DEFAULT_CLASSES))
    parser.add_argument("--subset-file", default="", help="JSON list of class names to include")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--frame-stride", type=int, default=2)
    # Removed --one-hand and --no-hand-presence flags to always use two hands and presence flags
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"Missing model file: {MODEL_PATH}")
        return 1

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Missing manifest: {manifest_path}")
        return 1

    items = load_manifest(manifest_path)
    if not items:
        print("Manifest is empty.")
        return 1

    label_names = None
    subset_signs = None
    subset_index = None
    if args.subset_file:
        subset_path = Path(args.subset_file)
    else:
        subset_path = None

    if subset_path is not None:
        if not subset_path.exists():
            print(f"Missing subset file: {subset_path}")
            return 1
        classes_path = Path(args.classes)
        if not classes_path.exists():
            print(f"Missing classes: {classes_path}")
            return 1
        label_names = load_label_names(classes_path)
        subset_signs = load_subset_signs(subset_path)
        if not subset_signs:
            print(f"Subset file is empty: {subset_path}")
            return 1
        subset_index = {name: idx for idx, name in enumerate(subset_signs)}

    neg_dir = Path(args.neg_dir)
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
    total_expected = total_items + total_negs
    kept = 0
    print(f"Starting extraction: {total_items} manifest clips + {total_negs} negatives")

    if tqdm is None:
        print("Tip: Install tqdm for progress bars: python -m pip install tqdm")


    manifest_iter = tqdm(items, desc="Manifest clips", unit="clip") if tqdm else items
    for item in manifest_iter:
        clip_path = Path(item["clip_path"])
        if label_names is not None and subset_signs is not None and subset_index is not None:
            label_name = label_names[int(item["label"])]
            if label_name not in subset_index:
                continue
            label_id = subset_index[label_name]
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
        if label_names is not None and subset_signs is not None and subset_index is not None:
            labels.append(int(label_id))
        else:
            labels.append(int(item["label"]))
        paths.append(str(clip_path))
        texts.append(item.get("text", ""))
        kept += 1

    if subset_signs is not None:
        nonsign_label = len(subset_signs)
    else:
        nonsign_label = max(labels) + 1 if labels else 0
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
    }
    labels_path = None
    if subset_signs is not None:
        labels_path = out_path.with_name(f"{out_path.stem}_labels.json")
        labels_path.write_text(
            json.dumps(subset_signs + ["Non-sign"], indent=2),
            encoding="utf-8",
        )
        meta["labels"] = str(labels_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved dataset: {out_path}")
    print(f"Metadata: {meta_path}")
    if labels_path is not None:
        print(f"Labels: {labels_path}")
    print(f"Samples: {meta['samples']} | Classes: {meta['classes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
