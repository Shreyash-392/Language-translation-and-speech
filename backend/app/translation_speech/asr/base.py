from abc import ABC, abstractmethod
import numpy as np


class ASRProvider(ABC):
    """Abstract Base Class for Speech Recognition Providers."""
    
    @abstractmethod
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000, language: str = "hi") -> str:
        """
        Transcribes audio data (1D numpy array, float32, 16kHz) to text.
        
        Args:
            audio_data: 1D float32 numpy array.
            sample_rate: Audio sampling rate (default 16000).
            language: Source language code (e.g., 'hi' for Hindi, 'sat' for Santali).
            
        Returns:
            Recognized text string.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if provider model is loaded and ready for inference."""
        pass
