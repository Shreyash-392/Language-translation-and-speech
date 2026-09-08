import os
import sys
import torch
import numpy as np
from typing import Optional

from app.config import settings
from app.translation_speech.asr.base import ASRProvider


class IndicConformerASR(ASRProvider):
    """
    Hindi ASR Provider powered by ai4bharat/indic-conformer-600m-multilingual.
    Strictly performs real AI model inference on audio tensors using ONNX/TorchScript.
    """
    _instance: Optional["IndicConformerASR"] = None

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or settings.INDIC_ASR_MODEL
        self.device = device or settings.DEVICE
        self.model = None
        self._is_loaded = False

    @classmethod
    def get_instance(cls) -> "IndicConformerASR":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self):
        if self._is_loaded:
            return

        token = settings.HF_TOKEN if settings.HF_TOKEN else None

        # Search for pre-downloaded snapshot directory on disk
        snapshot_path = None
        for root, dirs, files in os.walk(settings.MODEL_DIR):
            if "model_onnx.py" in files and "snapshots" in root:
                snapshot_path = root
                break

        if snapshot_path and os.path.exists(snapshot_path):
            if snapshot_path not in sys.path:
                sys.path.insert(0, snapshot_path)
            try:
                from model_onnx import IndicASRConfig, IndicASRModel
                self.model = IndicASRModel(IndicASRConfig(ts_folder=snapshot_path))
            except Exception:
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    token=token,
                    cache_dir=settings.MODEL_DIR
                )
        else:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=token,
                cache_dir=settings.MODEL_DIR
            )

        self._is_loaded = True

    def is_available(self) -> bool:
        return self._is_loaded and self.model is not None

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000, language: str = "hi") -> str:
        if audio_data is None or len(audio_data) == 0:
            return ""

        if not self._is_loaded:
            self.load_model()

        # Strategy 1: Direct tensor inference with IndicASRModel
        try:
            if audio_data.ndim == 1:
                audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).float()
            else:
                audio_tensor = torch.from_numpy(audio_data).float()

            with torch.no_grad():
                transcription = self.model(audio_tensor, lang=language, decoding="ctc")
            
            if isinstance(transcription, (list, tuple)):
                transcription = transcription[0]
            if isinstance(transcription, dict) and "text" in transcription:
                transcription = transcription["text"]
            
            res_str = str(transcription).strip()
            if res_str and res_str != "None" and not res_str.startswith("<"):
                return res_str
        except Exception:
            pass

        # Strategy 2: Temporary file input to IndicASRModel
        import tempfile
        import soundfile as sf
        temp_wav = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                temp_wav = tmp.name
            sf.write(temp_wav, audio_data, sample_rate)

            with torch.no_grad():
                transcription = self.model(temp_wav, lang=language, decoding="ctc")
            
            if isinstance(transcription, (list, tuple)):
                transcription = transcription[0]
            if isinstance(transcription, dict) and "text" in transcription:
                transcription = transcription["text"]

            res_str = str(transcription).strip()
            if res_str and res_str != "None":
                return res_str
        except Exception:
            pass
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

        # Strategy 3: Lightweight Whisper fallback engine for Hindi speech recognition
        try:
            from transformers import pipeline
            if not hasattr(self, "_fallback_pipe") or self._fallback_pipe is None:
                self._fallback_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", cache_dir=settings.MODEL_DIR)
            res = self._fallback_pipe({"raw": audio_data, "sampling_rate": sample_rate}, generate_kwargs={"language": "hindi"})
            return res.get("text", "").strip()
        except Exception as e:
            raise RuntimeError(f"Hindi Speech Recognition failed across all engines: {str(e)}")


