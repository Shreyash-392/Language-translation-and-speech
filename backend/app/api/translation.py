from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.translation_speech.translation.indictrans2 import IndicTrans2Translation
from app.translation_speech.benchmarking.metrics import Timer

router = APIRouter(prefix="/api/translation", tags=["Translation"])


class TranslationRequest(BaseModel):
    text: str = Field(..., example="नमस्ते मेरा नाम हर्षित है")
    source_language: str = Field(default="hin_Deva", example="hin_Deva")
    target_language: str = Field(default="sat_Olck", example="sat_Olck")


class TranslationResponse(BaseModel):
    source_language: str
    target_language: str
    source_text: str
    translated_text: str
    model: str
    translation_ms: float


@router.post("/hindi-to-santali", response_model=TranslationResponse)
async def translate_hindi_to_santali(request: TranslationRequest):
    """Translates Hindi text (hin_Deva) to Santali text (sat_Olck)."""
    return perform_translation(request.text, src_lang="hin_Deva", tgt_lang="sat_Olck")


@router.post("/santali-to-hindi", response_model=TranslationResponse)
async def translate_santali_to_hindi(request: TranslationRequest):
    """Translates Santali text (sat_Olck) to Hindi text (hin_Deva)."""
    return perform_translation(request.text, src_lang="sat_Olck", tgt_lang="hin_Deva")


@router.post("/bidirectional", response_model=TranslationResponse)
async def translate_bidirectional(request: TranslationRequest):
    """Translates text bidirectionally between Hindi (hin_Deva) and Santali (sat_Olck)."""
    return perform_translation(request.text, src_lang=request.source_language, tgt_lang=request.target_language)


def perform_translation(text: str, src_lang: str, tgt_lang: str) -> TranslationResponse:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    try:
        timer = Timer().start()
        translator = IndicTrans2Translation.get_instance()
        translated_text = translator.translate(
            text,
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )
        elapsed_ms = timer.stop()

        return TranslationResponse(
            source_language=src_lang,
            target_language=tgt_lang,
            source_text=text,
            translated_text=translated_text,
            model=translator.model_name,
            translation_ms=elapsed_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed ({src_lang} -> {tgt_lang}): {str(e)}")
