import os
from speech.config import SpeechConfig
from speech.pipeline import SpeechPipeline
from speech.tts import play_audio_file

def main():
    # You need these env vars for Google Cloud:
    # export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service_account.json"
    # And set your GCP Project ID:
    project_id = os.environ.get("GCP_PROJECT_ID", "")

    cfg = SpeechConfig(
        enable_translation=False, # change to true
        target_language="es",   # try "fr"
        enable_tts=False, # change to true 
        tts_language_code="en-US",
    )

    pipe = SpeechPipeline(cfg, project_id=project_id)

    # Simulate a stream of frame predictions:
    stream = (
        ["H"] * 8 + ["E"] * 8 + ["L"] * 8 + ["L"] * 8 + ["O"] * 8 +
        ["SPACE"] * 8 +
        ["W"] * 8 + ["O"] * 8 + ["R"] * 8 + ["L"] * 8 + ["D"] * 8 +
        ["SPACE"] * 8
    )

    last_audio = None
    for p in stream:
        out = pipe.update_frame_prediction(p)
        if out.confirmed_token:
            print(f"Confirmed: {out.confirmed_token} | EN: {out.english_text} | TR: {out.translated_text}")
        if out.tts_audio_path:
            last_audio = out.tts_audio_path

    if last_audio:
        print("Playing:", last_audio)
        play_audio_file(last_audio)

if __name__ == "__main__":
    main()
