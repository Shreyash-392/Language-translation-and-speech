import base64
from fastapi import APIRouter, UploadFile, File, HTTPException, Response
from pydantic import BaseModel

from app.translation_speech.pipeline.voice_translation import VoiceTranslationPipeline
from app.translation_speech.audio.preprocessing import AudioPreprocessingError

router = APIRouter(prefix="/api/translation-speech", tags=["Translation & Speech"])

pipeline = VoiceTranslationPipeline()


@router.post("/hindi-to-santali")
async def translate_speech_hindi_to_santali(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No audio file uploaded.")

    try:
        audio_bytes = await file.read()
        result = pipeline.process_voice_translation(audio_bytes)

        # Convert output WAV bytes to base64 data URI for easy playback in frontend
        wav_b64 = base64.b64encode(result["audio_wav_bytes"]).decode("utf-8") if result["audio_wav_bytes"] else ""
        audio_url = f"data:audio/wav;base64,{wav_b64}" if wav_b64 else ""

        return {
            "source_language": "hin_Deva",
            "target_language": "sat_Olck",
            "hindi_text": result["hindi_text"],
            "santali_text": result["santali_text"],
            "audio_url": audio_url,
            "latency_ms": result["latency_ms"],
            "system_metrics": result["system_metrics"],
            "pipeline_info": result["pipeline_info"]
        }
    except AudioPreprocessingError as ape:
        raise HTTPException(status_code=400, detail=f"Audio preprocessing error: {str(ape)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice translation pipeline failed: {str(e)}")


@router.post("/hindi-to-santali/audio-stream")
async def translate_speech_hindi_to_santali_stream(file: UploadFile = File(...)):
    """Returns raw audio/wav file response directly."""
    if not file:
        raise HTTPException(status_code=400, detail="No audio file uploaded.")

    try:
        audio_bytes = await file.read()
        result = pipeline.process_voice_translation(audio_bytes)

        return Response(
            content=result["audio_wav_bytes"],
            media_type="audio/wav",
            headers={
                "X-Hindi-Text": base64.b64encode(result["hindi_text"].encode("utf-8")).decode("utf-8"),
                "X-Santali-Text": base64.b64encode(result["santali_text"].encode("utf-8")).decode("utf-8"),
                "X-Total-Latency-MS": str(result["latency_ms"]["total_ms"])
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice translation failed: {str(e)}")
