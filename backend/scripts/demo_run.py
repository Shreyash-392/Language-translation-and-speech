import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.translation_speech import VernacularSpeechComponent

def main():
    print("============================================================")
    print("REAL AI INFERENCE TEST: IndicTrans2 Hindi -> Santali")
    print("============================================================")

    component = VernacularSpeechComponent(offline_mode=False)

    test_input = "नमस्ते, मेरा नाम श्रेयश है।"
    print(f"Input Hindi Text:  '{test_input}'")

    try:
        res = component.translate_text(test_input, source_lang="hin_Deva", target_lang="sat_Olck")
        print("\n[Real Model Output]")
        print(f"Santali Translated Text: '{res['translated_text']}'")
        print(f"Measured NMT Time:       {res['translation_ms']} ms")
    except Exception as e:
        print("\n[Execution Note]")
        print(f"{e}")
        print("\nTo download gated model weights online:")
        print("1. Visit https://huggingface.co/ai4bharat/indictrans2-indic-indic-dist-320M and click 'Agree and access repository'.")
        print("2. Add your HuggingFace user token to backend/.env: HF_TOKEN=hf_xxxx")
    print("============================================================")

if __name__ == "__main__":
    main()
