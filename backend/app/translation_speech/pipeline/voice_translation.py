import io
import wave
import numpy as np
from typing import Dict, Any, Tuple

from app.translation_speech.audio.preprocessing import preprocess_audio
from app.translation_speech.benchmarking.metrics import Timer, get_system_metrics
from app.translation_speech.asr.indic_conformer import IndicConformerASR
from app.translation_speech.translation.indictrans2 import IndicTrans2Translation
from app.translation_speech.tts.santali import SantaliTTS


class VoiceTranslationPipeline:
    """
    End-to-End Voice Translation Pipeline:
    Teacher Hindi Speech -> IndicConformer ASR -> Hindi Text -> IndicTrans2 -> Santali Text -> Santali TTS -> Student Santali Speech
    """

    def __init__(self):
        self.asr = IndicConformerASR.get_instance()
        self.translation = IndicTrans2Translation.get_instance()
        self.tts = SantaliTTS.get_instance()

    def process_voice_translation(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Processes raw audio payload through ASR -> NMT -> TTS pipeline with real-time latency measurement.
        """
        total_timer = Timer().start()
        
        # Step 1: Audio Preprocessing
        prep_timer = Timer().start()
        audio_data = preprocess_audio(audio_bytes, target_sample_rate=16000)
        preprocessing_ms = prep_timer.stop()

        # Step 2: Hindi Speech Recognition (ASR)
        asr_timer = Timer().start()
        is_asr_error = False
        try:
            hindi_text = self.asr.transcribe(audio_data, sample_rate=16000, language="hi")
        except Exception as e:
            hindi_text = f"[ASR Unavailable: {str(e)}]"
            is_asr_error = True
        asr_ms = asr_timer.stop()

        if not hindi_text:
            hindi_text = "[No clear speech recognized]"
            is_asr_error = True

        # Step 3: Neural Machine Translation (Hindi -> Santali)
        trans_timer = Timer().start()
        if not is_asr_error and not hindi_text.startswith("["):
            try:
                santali_text = self.translation.translate(hindi_text, src_lang="hin_Deva", tgt_lang="sat_Olck")
            except Exception as e:
                santali_text = f"[Translation Unavailable: {str(e)}]"
        else:
            santali_text = "[Translation skipped due to empty or unverified ASR speech]"
        translation_ms = trans_timer.stop()

        # Step 4: Santali Text-To-Speech (TTS)
        tts_timer = Timer().start()
        try:
            tts_prompt = santali_text if not santali_text.startswith("[") else "ᱡᱚᱦᱟᱨ"
            audio_arr, sr = self.tts.synthesize(tts_prompt, language="sat_Olck")
            wav_bytes = self._audio_array_to_wav_bytes(audio_arr, sr)
        except Exception as e:
            wav_bytes = b""
            sr = 16000
        tts_ms = tts_timer.stop()


        total_ms = total_timer.stop()
        metrics = get_system_metrics()

        # Trigger garbage collection to release temporary audio/tensor memory buffers
        try:
            import gc
            gc.collect()
        except Exception:
            pass

        return {
            "hindi_text": hindi_text,
            "santali_text": santali_text,
            "audio_wav_bytes": wav_bytes,
            "sample_rate": sr,
            "latency_ms": {
                "audio_preprocessing_ms": preprocessing_ms,
                "asr_ms": asr_ms,
                "translation_ms": translation_ms,
                "tts_ms": tts_ms,
                "total_ms": total_ms,
                "target_ms": 3000
            },
            "system_metrics": metrics,
            "pipeline_info": {
                "source_language": "hin_Deva",
                "target_language": "sat_Olck",
                "asr_model": self.asr.model_name,
                "translation_model": self.translation.model_name,
                "tts_model": self.tts.model_name,
                "reverse_pipeline_supported": False,
                "reverse_pipeline_status": "Santali ASR extension hook ready for future verification"
            }
        }

    def _audio_array_to_wav_bytes(self, audio_arr: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Converts float32 numpy audio array to 16-bit PCM WAV bytes."""
        audio_flat = audio_arr.flatten()
        audio_int16 = (np.clip(audio_flat, -1.0, 1.0) * 32767).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return buffer.getvalue()
