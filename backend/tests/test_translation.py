import pytest
from app.translation_speech.translation.base import TranslationProvider
from app.translation_speech.translation.indictrans2 import IndicTrans2Translation


class MockTranslationProvider(TranslationProvider):
    def translate(self, text: str, src_lang: str = "hin_Deva", tgt_lang: str = "sat_Olck") -> str:
        if not text:
            return ""
        if src_lang == "hin_Deva" and tgt_lang == "sat_Olck":
            return "ᱚᱞ ᱪᱤᱠᱤ ᱴᱮᱥᱴ"  # Mock Santali script output
        elif src_lang == "sat_Olck" and tgt_lang == "hin_Deva":
            return "नमस्ते मेरा नाम हर्षित है"  # Mock Hindi output
        return text

    def is_available(self) -> bool:
        return True


def test_mock_translation_provider_bidirectional():
    provider = MockTranslationProvider()
    assert provider.is_available() is True
    
    # Test Hindi to Santali
    sat_res = provider.translate("नमस्ते", src_lang="hin_Deva", tgt_lang="sat_Olck")
    assert sat_res == "ᱚᱞ ᱪᱤᱠᱤ ᱴᱮᱥᱴ"
    
    # Test Santali to Hindi
    hin_res = provider.translate("ᱚᱞ ᱪᱤᱠᱤ ᱴᱮᱥᱴ", src_lang="sat_Olck", tgt_lang="hin_Deva")
    assert hin_res == "नमस्ते मेरा नाम हर्षित है"


def test_indictrans2_lazy_loading():
    translator = IndicTrans2Translation.get_instance()
    assert translator is not None
    assert translator.model_name == "ai4bharat/indictrans2-indic-indic-dist-320M"
