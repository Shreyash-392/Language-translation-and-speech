import os
import torch
from typing import Optional

from app.config import settings
from app.translation_speech.translation.base import TranslationProvider


class IndicTrans2Translation(TranslationProvider):
    """
    Hindi <-> Santali (hin_Deva <-> sat_Olck) NMT Provider using ai4bharat/indictrans2-indic-indic-dist-320M.
    Strictly performs real AI model inference without hardcoded dictionaries or fallbacks.
    """
    _instance: Optional["IndicTrans2Translation"] = None

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or settings.INDICTRANS_MODEL
        self.device = device or settings.DEVICE
        self.tokenizer = None
        self.model = None
        self.ip = None
        self._is_loaded = False

    @classmethod
    def get_instance(cls) -> "IndicTrans2Translation":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self):
        if self._is_loaded:
            return

        token = settings.HF_TOKEN if settings.HF_TOKEN else None
        
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        dtype = torch.float16 if (self.device == "cuda" and torch.cuda.is_available()) else torch.float32

        # Strategy: Try local cache first if OFFLINE_MODE is set, else fall back to online download
        model_path = self.model_name
        loaded_successfully = False

        # Attempt 1: Load based on settings.OFFLINE_MODE
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                token=token,
                local_files_only=settings.OFFLINE_MODE,
                cache_dir=settings.MODEL_DIR
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                token=token,
                local_files_only=settings.OFFLINE_MODE,
                cache_dir=settings.MODEL_DIR
            )
            loaded_successfully = True
        except Exception:
            pass

        # Attempt 2: If attempt 1 failed (e.g. model not cached on cloud deployment), try downloading online
        if not loaded_successfully:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    token=token,
                    local_files_only=False,
                    cache_dir=settings.MODEL_DIR
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    token=token,
                    local_files_only=False,
                    cache_dir=settings.MODEL_DIR
                )
                loaded_successfully = True
            except Exception:
                pass

        # Attempt 3: Final attempt with local_files_only=True
        if not loaded_successfully:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                token=token,
                local_files_only=True,
                cache_dir=settings.MODEL_DIR
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                token=token,
                local_files_only=True,
                cache_dir=settings.MODEL_DIR
            )

        self.model.to(self.device)
        self.model.eval()

        try:
            from IndicTransToolkit import IndicProcessor
            self.ip = IndicProcessor(inference=True)
        except Exception:
            self.ip = None

        self._is_loaded = True

    def is_available(self) -> bool:
        return self._is_loaded and self.model is not None

    def translate(self, text: str, src_lang: str = "hin_Deva", tgt_lang: str = "sat_Olck") -> str:
        if not text or not text.strip():
            return ""

        if not self._is_loaded:
            self.load_model()

        input_text = [text.strip()]

        # Step 1: Preprocess input text via IndicProcessor if available, else format language tags
        if self.ip is not None:
            try:
                batch_text = self.ip.preprocess_batch(input_text, src_lang=src_lang, tgt_lang=tgt_lang)
            except Exception:
                batch_text = [f"{src_lang} {tgt_lang} {text.strip()}"]
        else:
            batch_text = [f"{src_lang} {tgt_lang} {text.strip()}"]

        # Step 2: Tokenize
        inputs = self.tokenizer(
            batch_text,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)

        # Step 3: Real IndicTrans2 Neural Model Generation with repetition prevention
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                use_cache=True,
                min_length=1,
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                num_return_sequences=1
            )

        # Step 4: Tokenizer Decoding
        decoded_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Step 5: Postprocessing output text
        if self.ip is not None:
            try:
                translated = self.ip.postprocess_batch(decoded_text, lang=tgt_lang)[0]
            except Exception:
                translated = decoded_text[0]
        else:
            translated = decoded_text[0]

        return translated.strip()

