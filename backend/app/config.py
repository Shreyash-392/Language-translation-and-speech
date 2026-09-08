import os
from pathlib import Path
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

# Enable SSL cert store on Windows if available
try:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    import pip_system_certs.wrapt_requests
except Exception:
    pass

# Base Directory of Backend and Project
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR.parent

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")

    HF_TOKEN: str = os.getenv("HF_TOKEN", "").strip()
    OFFLINE_MODE: bool = os.getenv("OFFLINE_MODE", "false").lower() in ("true", "1", "yes")
    
    # Auto-resolve model directory to whichever path contains cached models
    _default_model_dir = (
        str(PROJECT_DIR / "model_cache") 
        if (PROJECT_DIR / "model_cache").exists() 
        else str(BASE_DIR / "model_cache")
    )
    MODEL_DIR: str = os.getenv("MODEL_DIR", _default_model_dir)
    HF_HOME: str = os.getenv("HF_HOME", _default_model_dir)
    
    INDIC_ASR_MODEL: str = os.getenv("INDIC_ASR_MODEL", "ai4bharat/indic-conformer-600m-multilingual")
    INDICTRANS_MODEL: str = os.getenv("INDICTRANS_MODEL", "ai4bharat/indictrans2-indic-indic-dist-320M")
    TTS_MODEL: str = os.getenv("TTS_MODEL", "ai4bharat/indic-parler-tts")
    
    DEVICE: str = os.getenv("DEVICE") or ("cuda" if (lambda: __import__('torch').cuda.is_available())() else "cpu")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))

settings = Settings()

# Ensure MODEL_DIR path is absolute
settings.MODEL_DIR = os.path.abspath(settings.MODEL_DIR)
settings.HF_HOME = settings.MODEL_DIR

os.environ["HF_HOME"] = settings.HF_HOME
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

