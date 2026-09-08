import pytest
from app.translation_speech.benchmarking.metrics import Timer, get_system_metrics


def test_timer_benchmark():
    timer = Timer().start()
    sum_val = sum(range(100000))
    elapsed = timer.stop()
    
    assert elapsed >= 0.0
    assert isinstance(elapsed, float)


def test_system_metrics():
    metrics = get_system_metrics()
    assert isinstance(metrics, dict)
    assert "ram_usage_mb" in metrics
    assert "gpu_memory_allocated_mb" in metrics


def test_vernacular_speech_component_import():
    from app.translation_speech import VernacularSpeechComponent
    component = VernacularSpeechComponent(offline_mode=True)
    assert component is not None
    assert hasattr(component, "translate_voice")
    assert hasattr(component, "translate_text")
