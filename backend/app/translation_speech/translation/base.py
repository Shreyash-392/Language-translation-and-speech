from abc import ABC, abstractmethod


class TranslationProvider(ABC):
    """Abstract Base Class for Neural Machine Translation Providers."""

    @abstractmethod
    def translate(self, text: str, src_lang: str = "hin_Deva", tgt_lang: str = "sat_Olck") -> str:
        """
        Translates text from source language to target language code.
        
        Args:
            text: Input text string.
            src_lang: Source language code (e.g., 'hin_Deva').
            tgt_lang: Target language code (e.g., 'sat_Olck').
            
        Returns:
            Translated text string.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if provider model is loaded and ready."""
        pass
