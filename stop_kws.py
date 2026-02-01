import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import time
import os


class StopKWS:
    def __init__(self, model_path="models/stop_model.h5"):
        print("🧠 Загружаю нейросеть STOP...")
        
        if not os.path.exists(model_path):
            print(f"❌ Модель {model_path} не найдена!")
            print("   Сначала обучите модель: python train_stop.py")
            raise FileNotFoundError(f"Модель {model_path} не найдена")
        
        self.model = keras.models.load_model(model_path)
        self.last_trigger = 0
        self.cooldown = 1.2
        self.sample_rate = 16000
        print("✅ STOP модель загружена и готова")
    
    def extract_features(self, audio, sr=16000):
        """
        Извлечение 39 признаков из аудио:
        - 13 MFCC коэффициентов
        - 13 дельта-MFCC
        - 13 дельта-дельта-MFCC
        Итого: 39 признаков на кадр
        """
        if len(audio) == 0:
            return np.zeros((40, 39))
        
        # Конвертируем в float32 и нормализуем
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        # 1. MFCC коэффициенты (13 штук) - ТОЧНО КАК В ОБУЧЕНИИ!
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=13,  # БЫЛО 40, ДОЛЖНО БЫТЬ 13!
            n_fft=512,
            hop_length=256,
            n_mels=40
        )
        
        # 2. Дельта коэффициенты (ещё 13)
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # 3. Дельта-дельта коэффициенты (ещё 13)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 4. Объединяем все признаки: 13 + 13 + 13 = 39
        all_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        # Транспонируем: (кадры, 39 признаков)
        all_features = all_features.T
        
        # 5. Обрезаем/дополняем до 40 кадров
        if all_features.shape[0] < 40:
            pad = np.zeros((40 - all_features.shape[0], 39))
            all_features = np.vstack([all_features, pad])
        else:
            all_features = all_features[:40]
        
        return all_features
    
    def detect(self, audio, sr=16000):
        """
        Определение слова 'СТОП' с помощью нейросети
        Возвращает True если обнаружено слово 'стоп'
        """
        # Защита от слишком частых срабатываний
        now = time.time()
        if now - self.last_trigger < self.cooldown:
            return False
        
        # Проверка что звук достаточно громкий
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.01:  # Увеличил порог
            return False
        
        # Извлечение 39 признаков (как при обучении!)
        features = self.extract_features(audio, sr)
        
        # Подготовка для модели: (1, 40, 39)
        features = np.expand_dims(features, axis=0)
        
        # Предсказание нейросети
        prob = self.model.predict(features, verbose=0)[0][0]
        
        # Вывод информации для отладки
        print(f"[KWS] Вероятность 'СТОП': {prob:.3f} (RMS: {rms:.4f})")
        
        # Порог срабатывания - УВЕЛИЧИЛ!
        if prob > 0.95:  # БЫЛО 0.85, СТАЛО 0.95
            print(" ✅ СТОП ОБНАРУЖЕНО!")
            self.last_trigger = now
            return True
        else:
            return False