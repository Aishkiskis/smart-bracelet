from audio_processor import AudioProcessor
from stop_kws import StopKWS
from vibration_controller import VibrationController
import time


def main():
    audio = AudioProcessor()
    stop_detector = StopKWS()
    vibrator = VibrationController()

    print("\n🎧 НЕЙРОСЕТЕВОЙ STOP ЗАПУЩЕН (Ctrl+C для выхода)\n")

    try:
        while True:
            audio_data = audio.record(duration=0.8)

            if audio_data is None:
                continue

            if stop_detector.detect(audio_data):
                print("🛑 СТОП РАСПОЗНАН")
                vibrator.vibrate("STOP_PATTERN")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n🧹 Система остановлена")


if __name__ == "__main__":
    main()
