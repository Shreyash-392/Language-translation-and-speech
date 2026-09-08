import time
import psutil

try:
    import torch
except ImportError:
    torch = None


class Timer:
    """Precision wall-clock timer using time.perf_counter()."""
    def __init__(self):
        self._start_time = None
        self._end_time = None

    def start(self):
        self._start_time = time.perf_counter()
        self._end_time = None
        return self

    def stop(self) -> float:
        self._end_time = time.perf_counter()
        return self.elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time is not None else time.perf_counter()
        return round((end - self._start_time) * 1000.0, 2)


def get_system_metrics() -> dict:
    """Returns actual process RAM and CUDA GPU memory allocated in MB."""
    metrics = {
        "ram_usage_mb": None,
        "gpu_memory_allocated_mb": None,
        "gpu_memory_reserved_mb": None,
        "gpu_device_name": None
    }
    
    try:
        process = psutil.Process()
        metrics["ram_usage_mb"] = round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        pass

    if torch is not None and torch.cuda.is_available():
        try:
            metrics["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
            metrics["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
            metrics["gpu_device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

    return metrics
