from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class TTSProvider(ABC):
    """Abstract Base Class for Text-To-Speech Providers."""

    @abstractmethod
    def synthesize(self, text: str, language: str = "sat_Olck") -> Tuple[np.ndarray, int]:
        """
        Synthesizes audio array and sample rate from input text.
        
        Args:
            text: Input target text string.
            language: Target language code (default 'sat_Olck').
            
        Returns:
            Tuple of (1D float32 numpy audio array, sample_rate).
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if provider model is verified and ready for inference."""
        pass
