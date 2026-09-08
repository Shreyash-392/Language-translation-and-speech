import pytest
import numpy as np
import io
import wave
from app.translation_speech.audio.preprocessing import preprocess_audio, AudioPreprocessingError


def create_dummy_wav_bytes(duration_sec: float = 1.0, sample_rate: int = 44100, channels: int = 2) -> bytes:
    """Generates synthetic stereo WAV bytes for testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    if channels == 2:
        audio_data = np.vstack((sig, sig)).T
    else:
        audio_data = sig

    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    return buffer.getvalue()


def test_preprocess_valid_audio():
    wav_bytes = create_dummy_wav_bytes(duration_sec=1.0, sample_rate=44100, channels=2)
    processed_arr = preprocess_audio(wav_bytes, target_sample_rate=16000)
    
    assert isinstance(processed_arr, np.ndarray)
    assert processed_arr.ndim == 1  # Mono
    assert processed_arr.dtype == np.float32
    assert len(processed_arr) > 0


def test_preprocess_empty_audio():
    with pytest.raises(AudioPreprocessingError):
        preprocess_audio(b"")


def test_preprocess_corrupted_audio():
    with pytest.raises(AudioPreprocessingError):
        preprocess_audio(b"not an audio file payload")
