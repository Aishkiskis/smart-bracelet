import numpy as np


class KeywordSpotter:
    def __init__(self):
        pass

    def detect(self, audio, sample_rate=16000):
        if len(audio) == 0:
            return None

        # Энергия
        rms = np.sqrt(np.mean(audio ** 2))

        # Zero Crossing Rate
        zcr = np.mean(np.abs(np.diff(np.sign(audio))))

        # Частота (грубо, но стабильно)
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sample_rate)
        dominant_freq = freqs[np.argmax(fft)]

        duration = len(audio) / sample_rate

        print(
            f"[KS] rms={rms:.3f} zcr={zcr:.3f} "
            f"freq={dominant_freq:.0f}Hz dur={duration:.2f}s"
        )

        # ---------------- СТОП ----------------
        if (
            rms > 0.015 and
            zcr > 0.18 and
            duration < 0.9
        ):
            return "СТОП"

        # ---------------- ПОМОЩЬ ----------------
        if (
            rms > 0.015 and
            duration >= 0.7 and
            zcr < 0.28
        ):
            return "ПОМОЩЬ"

        # ---------------- ОПАСНО ----------------
        if (
            rms > 0.025 and
            duration >= 0.7 and
            zcr > 0.22
        ):
            return "ОПАСНО"

        return None
