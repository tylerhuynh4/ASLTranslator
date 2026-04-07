"""
CLI runner for live webcam model + speech pipeline integration.

Usage:
    python model_training/src/run_live_demo.py --camera 0 --enable-tts --play-audio
"""

from __future__ import annotations

import argparse
import threading

try:
    from inference.predictor import TemporalPredictor
except ModuleNotFoundError:
    from predictor import TemporalPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run webcam predictor with speech/text pipeline")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--speech-window", type=int, default=7)
    parser.add_argument("--speech-ratio", type=float, default=0.65)
    parser.add_argument("--enable-tts", action="store_true")
    parser.add_argument("--play-audio", action="store_true", help="Play TTS output locally when available")
    parser.add_argument(
        "--log-min-confidence",
        type=float,
        default=0.70,
        help="Only log confirmed tokens at/above this confidence",
    )
    parser.add_argument(
        "--log-cooldown-frames",
        type=int,
        default=20,
        help="Minimum frames before logging the same token again",
    )
    parser.add_argument(
        "--max-log-words",
        type=int,
        default=40,
        help="Maximum words to print to terminal (0 = unlimited)",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 = run until Ctrl+C")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        import cv2
    except Exception as exc:
        print(f"Missing OpenCV dependency: {exc}")
        return 2

    # Import speech modules lazily so plain predictor imports stay lightweight.
    play_audio_file = None
    try:
        from speech.config import SpeechConfig
        from speech.pipeline import SpeechPipeline
        if args.play_audio:
            from speech.tts import play_audio_file as _play_audio_file
            play_audio_file = _play_audio_file
    except Exception as exc:
        print(f"Unable to import speech pipeline: {exc}")
        return 2

    predictor = TemporalPredictor(
        threshold=args.threshold,
        min_frames=args.min_frames,
        smooth=args.smooth,
    )
    if predictor._init_error:
        print(f"Predictor initialization failed: {predictor._init_error}")
        return 1

    cfg = SpeechConfig(
        smooth_window=args.speech_window,
        stability_ratio=args.speech_ratio,
        enable_translation=False,
        enable_tts=args.enable_tts,
        target_language="en",
    )

    try:
        pipeline = SpeechPipeline(cfg)
    except Exception as exc:
        print(f"Speech pipeline initialization failed: {exc}")
        predictor.close()
        return 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera index {args.camera}")
        predictor.close()
        return 1

    print("Model + speech test started. Press 'q' or Ctrl+C to stop.")
    print("Live transcript:")
    print("-" * 60)

    frame_count = 0
    logged_words = 0
    log_limit_reached = False
    last_logged_token = None
    last_logged_frame = -10_000
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed; stopping.")
                break

            frame_count += 1
            display_frame = cv2.flip(frame, 1)
            label, conf = predictor.predict(frame)
            token = "" if label == predictor.non_sign_label else label
            out = pipeline.update_frame_prediction(token)

            # Draw prediction text on frame
            cv2.putText(
                display_frame,
                f"{label} ({conf:.2f}) (Speech: {args.enable_tts})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Live ASL Demo", display_frame)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

            if out.confirmed_token:
                should_log = True

                if conf < args.log_min_confidence:
                    should_log = False

                if (
                    should_log
                    and last_logged_token == out.confirmed_token
                    and frame_count - last_logged_frame < args.log_cooldown_frames
                ):
                    should_log = False

                if should_log and args.max_log_words > 0 and logged_words >= args.max_log_words:
                    should_log = False
                    if not log_limit_reached:
                        print("\n[Log limit reached; suppressing additional words]", flush=True)
                        log_limit_reached = True

                if should_log:
                    print(out.confirmed_token, end=" ", flush=True)
                    logged_words += 1
                    last_logged_token = out.confirmed_token
                    last_logged_frame = frame_count

            if out.tts_audio_path:
                print(f"\n[Audio: {out.tts_audio_path}]", end=" ")
                if play_audio_file is not None:
                    try:
                        threading.Thread(
                            target=play_audio_file,
                            args=(out.tts_audio_path,),
                            daemon=True,
                        ).start()
                    except Exception as exc:
                        print(f"\n[Audio playback failed: {exc}]", end=" ")

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        predictor.close()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
