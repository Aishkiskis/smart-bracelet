from faster_whisper import WhisperModel
import soundfile as sf
import tempfile
import os
import numpy as np


class WhisperASR:
    def __init__(self):
        print("🧠 Загружаю Whisper...")
        self.model = WhisperModel("base", device="cpu")
        print("✅ Whisper готов")

    def transcribe(self, audio: np.ndarray, sample_rate=16000) -> str:
        if len(audio) == 0:
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        sf.write(path, audio, sample_rate)

        segments, _ = self.model.transcribe(path)
        text = " ".join(s.text.strip() for s in segments)

        print(f"[WHISPER] «{text}»")

        os.remove(path)
        return text
