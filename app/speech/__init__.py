"""
Speech package for ASL Translator project.

Handles:
- Text processing (smoothing + token buffering)
- Translation (Google Translate API)
- Text-to-Speech (Google TTS)
- Full speech pipeline integration
"""

from .config import SpeechConfig
from .pipeline import SpeechPipeline, PipelineOutput
from .text_processing import TemporalSmoother, TokenBuffer, clean_text

__all__ = [
    "SpeechConfig",
    "SpeechPipeline",
    "PipelineOutput",
    "TemporalSmoother",
    "TokenBuffer",
    "clean_text",
]
