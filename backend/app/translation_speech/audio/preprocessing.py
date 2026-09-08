import io
import os
import tempfile
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None


class AudioPreprocessingError(Exception):
    """Custom exception raised when audio preprocessing fails."""
    pass


def preprocess_audio(audio_bytes: bytes, target_sample_rate: int = 16000) -> np.ndarray:
    """
    Normalizes incoming audio bytes to a 1D float32 numpy array sampled at target_sample_rate (16000 Hz, mono).
    
    Handles:
    - Stereo to mono conversion
    - Resampling to target_sample_rate
    - Empty or silent audio files
    - Invalid or corrupted format payloads
    """
    if not audio_bytes or len(audio_bytes) == 0:
        raise AudioPreprocessingError("Received empty audio payload.")

    audio_data = None
    sample_rate = None

    # Strategy 1: Try reading via soundfile
    if sf is not None:
        try:
            buffer = io.BytesIO(audio_bytes)
            data, sr = sf.read(buffer)
            if data is not None and len(data) > 0:
                audio_data, sample_rate = data, sr
        except Exception:
            pass

    # Strategy 2: Try reading via scipy.io.wavfile
    if audio_data is None:
        try:
            from scipy.io import wavfile
            buffer = io.BytesIO(audio_bytes)
            sr, data = wavfile.read(buffer)
            if data is not None and len(data) > 0:
                # Normalize integer formats to float32 (-1.0 to 1.0)
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                elif data.dtype == np.uint8:
                    data = (data.astype(np.float32) - 128.0) / 128.0
                audio_data, sample_rate = data, sr
        except Exception:
            pass

    # Strategy 3: Try reading via torchaudio
    if audio_data is None:
        try:
            import torchaudio
            buffer = io.BytesIO(audio_bytes)
            tensor, sr = torchaudio.load(buffer)
            if tensor is not None and tensor.numel() > 0:
                audio_data = tensor.numpy().T
                sample_rate = sr
        except Exception:
            pass

    # Strategy 4: Fallback to temporary file + librosa
    if audio_data is None and librosa is not None:
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                temp_file = tmp.name
            
            data, sr = librosa.load(temp_file, sr=target_sample_rate, mono=True)
            if data is not None and len(data) > 0:
                audio_data = data
                sample_rate = target_sample_rate
        except Exception as e:
            pass
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

    if audio_data is None:
        raise AudioPreprocessingError("Unsupported or corrupted audio format. Unable to process audio.")

    # Convert stereo / multi-channel to mono
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Ensure float32 format
    audio_data = audio_data.astype(np.float32)

    # Resample if necessary and librosa is available
    if sample_rate != target_sample_rate:
        if librosa is not None:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
        else:
            # Simple linear interpolation fallback if librosa is absent
            duration = len(audio_data) / sample_rate
            new_length = int(duration * target_sample_rate)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), new_length, endpoint=False),
                np.arange(len(audio_data)),
                audio_data
            ).astype(np.float32)

    # Silence and energy check
    max_amplitude = np.max(np.abs(audio_data))
    if max_amplitude < 1e-5:
        raise AudioPreprocessingError("Audio contains silence or very low energy signal.")

    return audio_data
