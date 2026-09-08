# AI-Powered Vernacular Pedagogy and Real-Time Translation Tool

## Module: Translation + Speech (Prototype Target: Santali)

An offline-capable AI speech translation pipeline built for primary education in mother-tongue languages. It converts Hindi teacher speech into Santali student speech using official open-source Indic models.

---

## 1. System Architecture

```text
Teacher Hindi Speech
        ↓
IndicConformer-600M-Multi ASR (ai4bharat/indic-conformer-600m-multilingual)
        ↓
Hindi Text (hin_Deva)
        ↓
IndicTrans2 Distilled 320M (ai4bharat/indictrans2-indic-indic-dist-320M)
        ↓
Santali Text (sat_Olck)
        ↓
Santali TTS (ai4bharat/indic-parler-tts)
        ↓
Student Santali Speech
```

### Modular Provider Structure
- **ASR Provider:** Abstract base `ASRProvider` → `IndicConformerASR`
- **Translation Provider:** Abstract base `TranslationProvider` → `IndicTrans2Translation`
- **TTS Provider:** Abstract base `TTSProvider` → `SantaliTTS`
- **Pipeline Orchestrator:** `VoiceTranslationPipeline`
- **Audio Normalizer:** 16 kHz Mono WAV audio preprocessor

---

## 2. Model Specifications & Evidence

| Task | Verified Model | Source Code | Target Code | Hardware Precision | Evidence / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hindi ASR** | `ai4bharat/indic-conformer-600m-multilingual` | `hi` (Hindi Audio) | `hin_Deva` Text | `torch.float16` | Officially verified IndicConformer ASR |
| **Translation** | `ai4bharat/indictrans2-indic-indic-dist-320M` | `hin_Deva` | `sat_Olck` | `torch.float16` | IndicTrans2 distilled 320M model |
| **Santali TTS** | `ai4bharat/indic-parler-tts` | `sat_Olck` | Santali Audio | `torch.float16` | Indic-Parler-TTS native Santali support |
| **Santali ASR** | Extension Hook | `sat_Olck` Audio | `sat_Olck` Text | N/A | Provider interface ready for future model verification |

> [!NOTE]
> **Santali ASR Status:** The reverse pipeline interface is architected for clean future integration once a Santali ASR model is verified.

---

## 3. Installation & Setup

### Prerequisites
- **Python:** Python 3.10 recommended (`py -3.10`)
- **GPU:** NVIDIA GPU with CUDA support (e.g., GTX 1650 / Tesla T4) or CPU mode.

### 1. Clone & Configure Environment
```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env`:
```env
HF_TOKEN=your_huggingface_token_optional
OFFLINE_MODE=false
MODEL_DIR=./model_cache
INDIC_ASR_MODEL=ai4bharat/indic-conformer-600m-multilingual
INDICTRANS_MODEL=ai4bharat/indictrans2-indic-indic-dist-320M
TTS_MODEL=ai4bharat/indic-parler-tts
DEVICE=cuda
```

### 2. Install Dependencies
```bash
py -3.10 -m pip install -r backend/requirements.txt
```

---

## 4. Pre-Downloading Models for Offline Mode

To run inference in offline classroom environments without internet dependency:

```bash
py -3.10 backend/scripts/download_models.py
```

After pre-downloading model weights into `MODEL_DIR`, enable offline mode in `.env`:
```env
OFFLINE_MODE=true
```

---

## 5. Running the Application

### Start Backend API & Web Interface
```bash
py -3.10 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the Web Application in your browser:
- **Interactive UI:** `http://localhost:8000`
- **Swagger API Docs:** `http://localhost:8000/docs`

---

## 6. API Reference

### 1. Direct Text Translation
`POST /api/translation/hindi-to-santali`
```json
// Request
{
  "text": "नमस्ते मेरा नाम हर्षित है",
  "source_language": "hin_Deva",
  "target_language": "sat_Olck"
}

// Response
{
  "source_language": "hin_Deva",
  "target_language": "sat_Olck",
  "source_text": "नमस्ते मेरा नाम हर्षित है",
  "translated_text": "ᱱᱚᱢᱚᱥᱛᱮ ᱤᱧᱟᱜ ᱧᱩᱛᱩᱢ ᱦᱚᱨᱥᱤᱛ ᱠᱟᱱᱟ",
  "model": "ai4bharat/indictrans2-indic-indic-dist-320M",
  "translation_ms": 142.5
}
```

### 2. Voice-to-Voice Pipeline
`POST /api/translation-speech/hindi-to-santali`
- **Input:** Multipart form upload (`file`: WAV/MP3 audio file)
- **Response:** JSON metadata, recognized Hindi text, translated Santali text, playable base64 audio URI, and detailed latency metrics (`asr_ms`, `translation_ms`, `tts_ms`, `total_ms`).

---

## 7. Latency & Resource Benchmarking

> [!IMPORTANT]
> **Performance Target:** Voice-to-voice latency target is **< 3000 ms**.
> Latency values presented by the application are **measured benchmark results**, not hardcoded targets.

### Benchmark Breakdown Response Structure
```json
{
  "latency_ms": {
    "audio_preprocessing_ms": 15.2,
    "asr_ms": 420.5,
    "translation_ms": 185.0,
    "tts_ms": 850.3,
    "total_ms": 1471.0,
    "target_ms": 3000
  },
  "system_metrics": {
    "ram_usage_mb": 1240.5,
    "gpu_memory_allocated_mb": 3120.0
  }
}
```

---

## 8. Mobile & Low-Cost Target Considerations
- **Development Runtime:** Tested on CUDA FP16 GPUs.
- **Low-Cost Target (Android 2GB RAM):** Provider interfaces support future replacement with quantized INT8 models, ONNX runtime, CTranslate2, or TFLite engines without rewriting application logic.

---

## 9. Running Tests
```bash
py -3.10 -m pytest backend/tests/ -v
```
