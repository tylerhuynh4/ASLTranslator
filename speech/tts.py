from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time

@dataclass
class TTSResult:
    audio_path: str
    text: str

class GoogleTTS:
    """
    Google Cloud Text-to-Speech.
    Requires GOOGLE_APPLICATION_CREDENTIALS env var for service account JSON.
    """
    def __init__(self, language_code="en-US", voice_name: str | None = None,
                 speaking_rate: float = 1.0, pitch: float = 0.0,
                 out_dir: str = "tts_out"):
        from google.cloud import texttospeech
        self.texttospeech = texttospeech
        self.client = texttospeech.TextToSpeechClient()

        self.language_code = language_code
        self.voice_name = voice_name
        self.speaking_rate = speaking_rate
        self.pitch = pitch

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

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
            audio_encoding=self.texttospeech.AudioEncoding.MP3,
            speaking_rate=self.speaking_rate,
            pitch=self.pitch,
        )

        response = self.client.synthesize_speech(
            input=input_text,
            voice=voice,
            audio_config=audio_config,
        )

        fname = f"tts_{int(time.time()*1000)}.mp3"
        path = self.out_dir / fname
        path.write_bytes(response.audio_content)
        return TTSResult(audio_path=str(path), text=text)

def play_audio_file(audio_path: str) -> None:
    """
    Simple local playback (works well on laptops).
    If your environment can't play audio, just skip calling this.
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
