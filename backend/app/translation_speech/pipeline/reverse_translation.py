from typing import Dict, Any
from app.translation_speech.translation.indictrans2 import IndicTrans2Translation
from app.translation_speech.benchmarking.metrics import Timer, get_system_metrics


class ReverseVoiceTranslationPipeline:
    """
    Reverse Pipeline:
    Student Santali Speech / Text -> Santali ASR (Hook) -> Santali Text (sat_Olck) -> IndicTrans2 -> Hindi Text (hin_Deva) -> Optional Hindi TTS
    """

    def __init__(self):
        self.translation = IndicTrans2Translation.get_instance()

    def process_santali_text_to_hindi(self, santali_text: str) -> Dict[str, Any]:
        """
        Translates Santali text (sat_Olck) to Hindi text (hin_Deva) with timing and resource tracking.
        """
        total_timer = Timer().start()

        trans_timer = Timer().start()
        hindi_text = self.translation.translate(
            santali_text,
            src_lang="sat_Olck",
            tgt_lang="hin_Deva"
        )
        translation_ms = trans_timer.stop()

        total_ms = total_timer.stop()
        metrics = get_system_metrics()

        return {
            "santali_text": santali_text,
            "hindi_text": hindi_text,
            "latency_ms": {
                "translation_ms": translation_ms,
                "total_ms": total_ms,
                "target_ms": 3000
            },
            "system_metrics": metrics,
            "pipeline_info": {
                "source_language": "sat_Olck",
                "target_language": "hin_Deva",
                "translation_model": self.translation.model_name,
                "santali_asr_status": "Extension hook pending verified pretrained Santali ASR model",
                "hindi_tts_status": "Optional Hindi TTS hook available"
            }
        }
