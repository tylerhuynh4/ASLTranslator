# speech/pipeline.py
from __future__ import annotations

from dataclasses import dataclass

from .config import SpeechConfig
from .text_processing import TemporalSmoother, TokenBuffer, clean_text


@dataclass
class PipelineOutput:
    confirmed_token: str | None
    english_text: str
    translated_text: str
    tts_audio_path: str | None


class SpeechPipeline:
    """
    Main pipeline:
      frame prediction -> smoothing -> token buffer -> optional translation -> optional TTS
    """

    def __init__(self, cfg: SpeechConfig, project_id: str | None = None):
        self.cfg = cfg
        self.smoother = TemporalSmoother(cfg.smooth_window, cfg.stability_ratio)
        self.token_buffer = TokenBuffer(cfg.space_token, cfg.delete_token, cfg.clear_token)

        self._frame_since_confirm = 999
        self._last_confirmed: str | None = None  # debounce repeats

        self.translator = None
        self.tts = None

        if cfg.enable_translation:
            if not project_id:
                raise ValueError("project_id is required when enable_translation=True")
            from .translator import GoogleTranslator
            self.translator = GoogleTranslator(project_id=project_id)

        if cfg.enable_tts:
            from .tts import GoogleTTS
            self.tts = GoogleTTS(
                language_code=cfg.tts_language_code,
                voice_name=cfg.tts_voice_name,
                speaking_rate=cfg.tts_speaking_rate,
                pitch=cfg.tts_pitch,
            )

    def reset(self) -> None:
        self.token_buffer.reset()
        self._frame_since_confirm = 999
        self._last_confirmed = None

    def update_frame_prediction(self, pred: str) -> PipelineOutput:
        """
        Call this every frame with the model's predicted label (e.g., 'H', 'SPACE').
        Returns updated pipeline outputs.
        """
        self._frame_since_confirm += 1

        confirmed = self.smoother.update(pred)

        # Debounce: block repeats only if they happen too soon (prevents spam but allows "LL")
        if confirmed is not None and confirmed == self._last_confirmed:
            if self._frame_since_confirm < self.cfg.repeat_delay_frames:
                confirmed = None

        # Avoid confirming too frequently (gap-based)
        if confirmed and self._frame_since_confirm < self.cfg.min_confirm_gap_frames:
            confirmed = None

        if confirmed:
            self.token_buffer.add_token(confirmed)
            self._frame_since_confirm = 0
            self._last_confirmed = confirmed

        # Construct English text (always compute, even if nothing confirmed this frame)
        raw_text = (self.token_buffer.state.sentence + self.token_buffer.state.current_word).strip()
        english = clean_text(raw_text)

        translated = english
        if (
            self.cfg.enable_translation
            and self.translator
            and self.cfg.target_language
            and self.cfg.target_language != "en"
            and english
        ):
            translated = self.translator.translate_text(english, target_language=self.cfg.target_language)

        audio_path = None
        # Only speak when SPACE is confirmed (word boundary)
        if self.cfg.enable_tts and self.tts and confirmed == self.cfg.space_token and english:
            res = self.tts.synthesize(english)
            audio_path = res.audio_path

        return PipelineOutput(
            confirmed_token=confirmed,
            english_text=english,
            translated_text=translated,
            tts_audio_path=audio_path,
        )

 