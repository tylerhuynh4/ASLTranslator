from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TTSResult:
    audio_bytes: bytes
    sample_rate_hz: int
    text: str

class GoogleTTS:
    """
    Google Cloud Text-to-Speech.
    Requires GOOGLE_APPLICATION_CREDENTIALS env var for service account JSON.
    """
    def __init__(self, language_code="en-US", voice_name: str | None = None,
                 speaking_rate: float = 1.0, pitch: float = 0.0,
                 sample_rate_hz: int = 24000):
        from google.cloud import texttospeech
        self.texttospeech = texttospeech
        self.client = texttospeech.TextToSpeechClient()

        self.language_code = language_code
        self.voice_name = voice_name
        self.speaking_rate = speaking_rate
        self.pitch = pitch
        self.sample_rate_hz = int(sample_rate_hz)

    def synthesize(self, text: str) -> TTSResult:
        if not text:
            raise ValueError("TTS text is empty")

        input_text = self.texttospeech.SynthesisInput(text=text)

        if self.voice_name:
            voice = self.texttospeech.VoiceSelectionParams(
                language_code=self.language_code,
                name=self.voice_name,
            )
        else:
            voice = self.texttospeech.VoiceSelectionParams(
                language_code=self.language_code,
                ssml_gender=self.texttospeech.SsmlVoiceGender.NEUTRAL,
            )

        audio_config = self.texttospeech.AudioConfig(
            audio_encoding=self.texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=self.speaking_rate,
            pitch=self.pitch,
            sample_rate_hertz=self.sample_rate_hz,
        )

        response = self.client.synthesize_speech(
            input=input_text,
            voice=voice,
            audio_config=audio_config,
        )

        return TTSResult(
            audio_bytes=response.audio_content,
            sample_rate_hz=self.sample_rate_hz,
            text=text,
        )


def play_audio_bytes(audio_bytes: bytes, sample_rate_hz: int = 24000) -> None:
    """
    Play LINEAR16 mono PCM audio bytes directly from memory.
    """
    import simpleaudio as sa

    play_obj = sa.play_buffer(
        audio_bytes,
        num_channels=1,
        bytes_per_sample=2,
        sample_rate=int(sample_rate_hz),
    )
    play_obj.wait_done()

def play_audio_file(audio_path: str) -> None:
    """
    Backward-compatible file playback helper (MP3/WAV/etc.).
    """
    from pydub import AudioSegment
    import simpleaudio as sa

    seg = AudioSegment.from_file(audio_path)
    play_obj = sa.play_buffer(
        seg.raw_data,
        num_channels=seg.channels,
        bytes_per_sample=seg.sample_width,
        sample_rate=seg.frame_rate,
    )
    play_obj.wait_done()
