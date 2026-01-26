import numpy as np


class StopDetector:
    def __init__(self):
        pass

    def detect(self, audio, sample_rate=16000):
        if len(audio) == 0:
            return False

        rms = np.sqrt(np.mean(audio ** 2))
        zcr = np.mean(np.abs(np.diff(np.sign(audio))))
        duration = len(audio) / sample_rate

        # 🔍 Диагностика — ТОЛЬКО если звук есть
        if rms > 0.01:
            print(f"[STOP] rms={rms:.3f} zcr={zcr:.3f} dur={duration:.2f}s")

        # ❌ AUDIO GATE — тишину и шум отсекаем
        if rms < 0.015:
            return False

        # ❌ обычная речь длиннее
        if duration > 0.8:
            return False

        # ❌ слишком плавно — не «стоп»
        if zcr < 0.18:
            return False

        # ✅ короткое + резкое + громкое
        return True
