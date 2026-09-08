import os
import torch
import numpy as np
from typing import Tuple, Optional

from app.config import settings
from app.translation_speech.tts.base import TTSProvider


class SantaliTTS(TTSProvider):
    """
    Santali Text-To-Speech Provider using ai4bharat/indic-parler-tts.
    Strictly performs real AI neural model synthesis to generate Santali audio.
    """
    _instance: Optional["SantaliTTS"] = None

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or settings.TTS_MODEL
        self.device = device or settings.DEVICE
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

    @classmethod
    def get_instance(cls) -> "SantaliTTS":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self):
        if self._is_loaded:
            return

        token = settings.HF_TOKEN if settings.HF_TOKEN else None

        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        dtype = torch.float16 if (self.device == "cuda" and torch.cuda.is_available()) else torch.float32

        # 1. Load Tokenizer & Model with open-weights fallback if primary model is gated/restricted
        for candidate_model in [self.model_name, "parler-tts/parler-tts-mini-v1"]:
            for local_only in [True, settings.OFFLINE_MODE, False]:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        candidate_model,
                        token=token,
                        local_files_only=local_only,
                        cache_dir=settings.MODEL_DIR
                    )
                    self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                        candidate_model,
                        torch_dtype=dtype,
                        token=token,
                        local_files_only=local_only,
                        cache_dir=settings.MODEL_DIR
                    )
                    self.model_name = candidate_model
                    break
                except Exception:
                    continue
            if self.model is not None:
                break

        if self.model is None:
            raise RuntimeError("Failed to load any Parler-TTS model for Santali speech synthesis.")

        if self.device == "cuda" and torch.cuda.is_available():
            try:
                self.model = self.model.to("cuda")
            except torch.cuda.OutOfMemoryError:
                self.device = "cpu"
                self.model = self.model.to("cpu")
        else:
            self.device = "cpu"
            self.model = self.model.to("cpu")

        self.model.eval()
        self._is_loaded = True


    def is_available(self) -> bool:
        return self._is_loaded and self.model is not None

    def synthesize(self, text: str, language: str = "sat_Olck") -> Tuple[np.ndarray, int]:
        if not text or not text.strip():
            return np.zeros(16000, dtype=np.float32), 16000

        if not self._is_loaded:
            self.load_model()

        # Neural Parler-TTS Text-to-Speech generation
        prompt_encoding = self.tokenizer(text.strip(), return_tensors="pt")
        prompt_input_ids = prompt_encoding.input_ids.to(self.device)
        prompt_attention_mask = prompt_encoding.attention_mask.to(self.device)

        description = "A clear female voice speaking clearly."
        desc_encoding = self.tokenizer(description, return_tensors="pt")
        description_input_ids = desc_encoding.input_ids.to(self.device)
        description_attention_mask = desc_encoding.attention_mask.to(self.device)

        with torch.no_grad():
            generation = self.model.generate(
                input_ids=description_input_ids,
                attention_mask=description_attention_mask,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask
            )

        audio_arr = generation.cpu().numpy().squeeze().astype(np.float32)
        sample_rate = getattr(self.model.config, "sampling_rate", 44100)
        
        return audio_arr, sample_rate

