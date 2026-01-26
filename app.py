import time
import numpy as np
import re

from audio_processor import AudioProcessor
from whisper_asr import WhisperASR
from vibration_controller import VibrationController


CHUNK_SECONDS = 0.4

SILENCE_THRESHOLD = 0.002   # 🔥 ниже для женского голоса
SILENCE_TIME = 0.6          # пауза = конец фразы
COOLDOWN = 2.0


def clean_text(text: str) -> list[str]:
    """
    убираем пунктуацию и приводим к словам
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def main():
    audio = AudioProcessor()
    whisper = WhisperASR()
    vibrator = VibrationController()

    buffer = np.array([], dtype=np.float32)
    silence_start = None
    last_trigger = 0

    print("\n🎧 СИСТЕМА ЗАПУЩЕНА — говорите\n")

    while True:
        chunk = audio.record(CHUNK_SECONDS)
        buffer = np.concatenate([buffer, chunk])

        rms = np.sqrt(np.mean(chunk ** 2))
        print(f"[AUDIO] rms={rms:.4f}")

        if rms < SILENCE_THRESHOLD:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= SILENCE_TIME:
                # 🧠 фраза закончилась
                if len(buffer) > 0:
                    text = whisper.transcribe(buffer)
                    buffer = np.array([], dtype=np.float32)
                    silence_start = None

                    words = clean_text(text)
                    last_words = words[-3:]

                    print(f"[DEBUG] последние слова: {last_words}")

                    if "стоп" in last_words:
                        now = time.time()
                        if now - last_trigger > COOLDOWN:
                            print("🛑 СЛОВО «СТОП» В КОНЦЕ ФРАЗЫ")
                            vibrator.vibrate("STOP_PATTERN")
                            last_trigger = now
                else:
                    buffer = np.array([], dtype=np.float32)
                    silence_start = None
        else:
            silence_start = None


if __name__ == "__main__":
    main()
