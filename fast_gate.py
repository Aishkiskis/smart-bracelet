import numpy as np


class FastGate:
    def is_candidate(self, audio, sample_rate=16000):
        rms = np.sqrt(np.mean(audio ** 2))
        zcr = np.mean(np.abs(np.diff(np.sign(audio))))
        duration = len(audio) / sample_rate

        if rms < 0.02:
            return False

        if duration > 0.8:
            return False

        if zcr < 0.18:
            return False

        print(f"[GATE] candidate rms={rms:.3f} zcr={zcr:.3f}")
        return True
