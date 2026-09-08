import urllib.request
import json
import io
import wave
import numpy as np
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def test_api():
    # 1. Test Text Translation (Hindi -> Santali)
    req1 = urllib.request.Request(
        "http://127.0.0.1:8000/api/translation/hindi-to-santali",
        data=json.dumps({"text": "नमस्ते मेरा नाम हर्षित है"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res1 = json.loads(urllib.request.urlopen(req1).read().decode("utf-8"))
    print("1. Hindi -> Santali Text API:")
    print(f"   Source: '{res1['source_text']}'")
    print(f"   Target: '{res1['translated_text']}' ({res1['target_language']})")
    print(f"   Time:   {res1['translation_ms']} ms\n")

    # 2. Test Text Translation (Santali -> Hindi)
    req2 = urllib.request.Request(
        "http://127.0.0.1:8000/api/translation/santali-to-hindi",
        data=json.dumps({"text": "ᱱᱚᱢᱚᱥᱛᱮ ᱤᱧᱟᱜ ᱧᱩᱛᱩᱢ ᱦᱚᱨᱥᱤᱛ ᱠᱟᱱᱟ"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res2 = json.loads(urllib.request.urlopen(req2).read().decode("utf-8"))
    print("2. Santali -> Hindi Text API:")
    print(f"   Source: '{res2['source_text']}'")
    print(f"   Target: '{res2['translated_text']}' ({res2['target_language']})")
    print(f"   Time:   {res2['translation_ms']} ms\n")

    # 3. Test Voice Pipeline API (Hindi Audio -> Santali Speech)
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(sig.tobytes())
    wav_bytes = buf.getvalue()

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="test.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8") + wav_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req3 = urllib.request.Request(
        "http://127.0.0.1:8000/api/translation-speech/hindi-to-santali",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    res3 = json.loads(urllib.request.urlopen(req3).read().decode("utf-8"))
    print("3. Voice Pipeline API:")
    print(f"   ASR Hindi:  '{res3['hindi_text']}'")
    print(f"   NMT Santali: '{res3['santali_text']}'")
    print(f"   Audio URL:  {res3['audio_url'][:40]}... (Base64 WAV payload)")
    print(f"   Latency:    {res3['latency_ms']['total_ms']} ms\n")

if __name__ == "__main__":
    test_api()
