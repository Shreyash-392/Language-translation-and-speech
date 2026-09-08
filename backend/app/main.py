import os
import sys
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

# Add parent 'backend' directory to sys.path programmatically
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.api.translation import router as translation_router
from app.api.translation_speech import router as speech_router

logger = logging.getLogger("uvicorn")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lazy model loading on-demand to conserve RAM on cloud deployments
    yield

app = FastAPI(
    title="Vernacular Pedagogy Translation & Speech API",
    description="AI-Powered Vernacular Pedagogy and Real-Time Translation Tool (Hindi to Santali sat_Olck)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for development & deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(translation_router)
app.include_router(speech_router)

# Mount Frontend Static Files directory
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {
            "status": "online",
            "message": "Vernacular Pedagogy API is running.",
            "docs_url": "/docs"
        }
else:
    @app.get("/")
    async def serve_root():
        return {
            "status": "online",
            "message": "Vernacular Pedagogy API is running.",
            "docs_url": "/docs"
        }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "offline_mode": settings.OFFLINE_MODE,
        "device": settings.DEVICE,
        "indic_asr_model": settings.INDIC_ASR_MODEL,
        "indictrans_model": settings.INDICTRANS_MODEL,
        "tts_model": settings.TTS_MODEL
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
