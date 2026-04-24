from pathlib import Path
from predictor import TemporalPredictor

# all paths relative to app.py

_HERE = Path(__file__).resolve().parent

MODEL_PATH = _HERE / "model" / "demo_model_2hand.pt"
SCALER_PATH = _HERE / "model" / "demo_scaler_2hand.npz"
LABELS_PATH = _HERE / "model" / "demo_labels_2hand.json"
HAND_TASK_PATH = _HERE / "model_training" / "hand_landmarker.task"
POSE_TASK_PATH = _HERE / "model_training" / "pose_landmarker_full.task"

# Create + returns ready for use temporal predictor
def load_predictor(threshold: float = 0.35, # min softmax confidence to accept a prediction (0-1)
                   min_frames: int = 8,     # frames buffer must fill b4 inference begins
                   smooth: int = 5,) -> TemporalPredictor:  # smooth is how many probability vectors to avg (reduce flickering on ambiguous frames)
    predictor = TemporalPredictor(model_path = MODEL_PATH,
                                  scaler_path = SCALER_PATH,
                                  labels_path = LABELS_PATH,
                                  hand_model_path = HAND_TASK_PATH,
                                  pose_model_path = POSE_TASK_PATH,
                                  threshold = threshold,
                                  min_frames = min_frames,
                                  smooth = smooth,
                                  device = "auto",)
   
    predictor._initialize() # Initializes early for error detection
    if predictor._init_error:
        raise RuntimeError(f"Failed to initialize: {predictor._init_error}")
    return predictor 