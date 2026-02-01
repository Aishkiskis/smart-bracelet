import sounddevice as sd
import numpy as np


class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def record(self, duration=1.0):
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        audio = audio.flatten()

        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        ratio = peak / max(rms, 0.001)  # избегаем деления на 0

        print(f"[AUDIO] rms={rms:.4f} peak={peak:.4f} ratio={ratio:.1f}x")

        # ❌ слишком тихо
        if rms < 0.01:
            print("   [FILTER] слишком тихо")
            return None

        # ❌ импульсный шум (стук)
        if ratio > 10 and rms < 0.1:
            print(f"   [FILTER] импульсный шум (ratio={ratio:.1f}x)")
            return None

        return audio