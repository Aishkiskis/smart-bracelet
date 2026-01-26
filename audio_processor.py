import sounddevice as sd
import numpy as np


class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def record(self, duration=0.5):
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        audio = audio.flatten()
        rms = np.sqrt(np.mean(audio ** 2)) if len(audio) else 0.0
        print(f"[AUDIO] получено {len(audio)} сэмплов | rms={rms:.4f}")

        return audio
