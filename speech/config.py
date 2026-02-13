from dataclasses import dataclass

@dataclass
class SpeechConfig:
    # Smoothing
    smooth_window: int = 7              # frames in buffer
    stability_ratio: float = 0.65       # most-common must be >= 65% of window
    min_confirm_gap_frames: int = 3     # avoid re-confirming same token too fast
    repeat_delay_frames: int = 10       # allow same token again after 10 frames 

    # Tokenization
    space_token: str = "SPACE"
    delete_token: str = "DEL"
    clear_token: str = "CLEAR"

    # Translation
    enable_translation: bool = True
    target_language: str = "en"         # set to "es", "fr", etc.

    # TTS
    enable_tts: bool = True
    tts_language_code: str = "en-US"
    tts_voice_name: str | None = None   # e.g. "en-US-Neural2-C" if you want
    tts_speaking_rate: float = 1.0
    tts_pitch: float = 0.0
