import os
import sys

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings

def main():
    print("=" * 60)
    print("AI-Powered Vernacular Pedagogy Model Downloader")
    print(f"Target Cache Directory: {settings.MODEL_DIR}")
    print("=" * 60)

    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    os.environ["HF_HOME"] = settings.MODEL_DIR

    token = settings.HF_TOKEN if settings.HF_TOKEN else None

    # 1. Download IndicConformer ASR Model
    print(f"\n[1/3] Pre-downloading Hindi ASR Model: {settings.INDIC_ASR_MODEL}...")
    try:
        from transformers import AutoModel
        AutoModel.from_pretrained(settings.INDIC_ASR_MODEL, trust_remote_code=True, token=token, cache_dir=settings.MODEL_DIR)
        print("✓ ASR Model downloaded successfully.")
    except Exception as e:
        print(f"✗ Failed to download ASR Model: {e}")

    # 2. Download IndicTrans2 Translation Model
    print(f"\n[2/3] Pre-downloading Translation Model: {settings.INDICTRANS_MODEL}...")
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        AutoTokenizer.from_pretrained(settings.INDICTRANS_MODEL, token=token, trust_remote_code=True, cache_dir=settings.MODEL_DIR)
        AutoModelForSeq2SeqLM.from_pretrained(settings.INDICTRANS_MODEL, token=token, trust_remote_code=True, cache_dir=settings.MODEL_DIR)
        print("✓ IndicTrans2 Model downloaded successfully.")
    except Exception as e:
        print(f"✗ Failed to download IndicTrans2 Model: {e}")

    # 3. Download Indic Parler TTS Model
    print(f"\n[3/3] Pre-downloading Santali TTS Model: {settings.TTS_MODEL}...")
    try:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        ParlerTTSForConditionalGeneration.from_pretrained(settings.TTS_MODEL, token=token, cache_dir=settings.MODEL_DIR)
        AutoTokenizer.from_pretrained(settings.TTS_MODEL, token=token, cache_dir=settings.MODEL_DIR)
        print("✓ Santali TTS Model downloaded successfully.")
    except Exception as e:
        print(f"✗ Failed to download TTS Model: {e}")

    print("\n" + "=" * 60)
    print("Model Download Process Completed.")
    print("You can now set OFFLINE_MODE=true in .env for local inference.")
    print("=" * 60)

if __name__ == "__main__":
    main()
