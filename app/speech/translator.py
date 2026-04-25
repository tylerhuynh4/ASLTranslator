from __future__ import annotations

class GoogleTranslator:
    """
    Uses Google Cloud Translate v3.
    Requires GOOGLE_APPLICATION_CREDENTIALS env var to point to your service account JSON.
    Also requires GCP project ID.
    """
    def __init__(self, project_id: str, location: str = "global"):
        from google.cloud import translate_v3 as translate
        self.translate = translate.TranslationServiceClient()
        self.parent = f"projects/{project_id}/locations/{location}"

    def translate_text(self, text: str, target_language: str, source_language: str | None = None) -> str:
        if not text:
            return ""
        req = {
            "parent": self.parent,
            "contents": [text],
            "target_language_code": target_language,
            "mime_type": "text/plain",
        }
        if source_language:
            req["source_language_code"] = source_language

        resp = self.translate.translate_text(request=req)
        return resp.translations[0].translated_text if resp.translations else text
