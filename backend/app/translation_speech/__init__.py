"""
Vernacular Speech & Translation Component
Offline-First Module for Mother Tongue-Based Primary Education (Hindi ↔ Santali sat_Olck)

Supports:
1. Hindi Teacher Speech -> Hindi ASR -> IndicTrans2 -> Santali sat_Olck -> Santali Speech
2. Bidirectional Text Translation (Hindi hin_Deva ↔ Santali sat_Olck)
3. Reverse Pipeline Extension Hooks (Santali -> Hindi)
"""

import os
from typing import Dict, Any, Optional

from app.translation_speech.pipeline.voice_translation import VoiceTranslationPipeline
from app.translation_speech.pipeline.reverse_translation import ReverseVoiceTranslationPipeline
from app.translation_speech.audio.preprocessing import preprocess_audio, AudioPreprocessingError
from app.translation_speech.benchmarking.metrics import Timer, get_system_metrics


class VernacularSpeechComponent:
    """
    Plug-and-play offline component for embedding Vernacular Pedagogy Translation & Speech
    into larger Python applications or hosting as an independent service.
    """

    def __init__(self, offline_mode: Optional[bool] = None, model_dir: Optional[str] = None):
        if offline_mode is not None:
            os.environ["OFFLINE_MODE"] = "true" if offline_mode else "false"
        if model_dir:
            os.environ["MODEL_DIR"] = model_dir
            os.environ["HF_HOME"] = model_dir

        self.pipeline = VoiceTranslationPipeline()
        self.reverse_pipeline = ReverseVoiceTranslationPipeline()

    def translate_voice(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Translates Hindi teacher audio payload into Santali text and speech."""
        return self.pipeline.process_voice_translation(audio_bytes)

    def translate_text(self, text: str, source_lang: str = "hin_Deva", target_lang: str = "sat_Olck") -> Dict[str, Any]:
        """Translates text bidirectionally between Hindi (hin_Deva) and Santali (sat_Olck)."""
        timer = Timer().start()
        translated_text = self.pipeline.translation.translate(
            text,
            src_lang=source_lang,
            tgt_lang=target_lang
        )
        elapsed_ms = timer.stop()

        return {
            "source_language": source_lang,
            "target_language": target_lang,
            "source_text": text,
            "translated_text": translated_text,
            "translation_ms": elapsed_ms
        }

    def translate_santali_to_hindi(self, santali_text: str) -> Dict[str, Any]:
        """Translates Santali text (sat_Olck) to Hindi text (hin_Deva)."""
        return self.reverse_pipeline.process_santali_text_to_hindi(santali_text)


__all__ = [
    "VernacularSpeechComponent",
    "VoiceTranslationPipeline",
    "ReverseVoiceTranslationPipeline",
    "AudioPreprocessingError"
]
