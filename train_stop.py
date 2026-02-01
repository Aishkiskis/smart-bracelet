# train_stop.py — ОБУЧЕНИЕ НА 150 ФАЙЛАХ
import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA_DIR = "train_stop"
SR = 16000
N_MFCC = 13
MAX_LEN = 40

print("="*60)
print("   🧠 ОБУЧЕНИЕ НЕЙРОСЕТИ НА 150 ФАЙЛАХ")
print("="*60)

def extract_mfcc(path):
    """Извлечение MFCC признаков с улучшенной обработкой"""
    try:
        audio, sr = librosa.load(path, sr=SR)
        
        # Нормализация громкости
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        
        # MFCC с улучшенными параметрами
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=512,
            hop_length=256,
            n_mels=40
        )
        
        # Дельта и дельта-дельта коэффициенты (улучшают точность)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Объединяем все признаки
        all_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        all_features = all_features.T  # (кадры, 39 признаков)
        
        # Обрезаем/дополняем
        if all_features.shape[0] < MAX_LEN:
            pad = np.zeros((MAX_LEN - all_features.shape[0], 39))
            all_features = np.vstack([all_features, pad])
        else:
            all_features = all_features[:MAX_LEN]
            
        return all_features
        
    except Exception as e:
        print(f"⚠️ Ошибка обработки {path}: {e}")
        return None

# Загрузка данных
X = []
y = []

print("📦 Загружаю данные...")

# Считаем файлы
stop_count = 0
other_count = 0

for label, folder in [(1, "stop"), (0, "other")]:
    folder_path = os.path.join(DATA_DIR, folder)
    
    if not os.path.exists(folder_path):
        print(f"❌ Папка {folder_path} не найдена!")
        continue
    
    files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    print(f"   📁 {folder}: {len(files)} файлов")
    
    for file in files:
        path = os.path.join(folder_path, file)
        features = extract_mfcc(path)
        
        if features is not None:
            X.append(features)
            y.append(label)
            
            if label == 1:
                stop_count += 1
            else:
                other_count += 1

X = np.array(X)
y = np.array(y)

print(f"\n✅ Загружено {len(X)} файлов:")
print(f"   🔴 'СТОП': {stop_count} файлов")
print(f"   🔵 'Других': {other_count} файлов")
print(f"   📊 Всего: {len(X)} примеров")

if len(X) < 100:
    print(f"\n⚠️ Мало данных! Нужно минимум 100, а есть {len(X)}")
    print("   Добавьте больше записей через record_better.py")
    exit()

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Разделение данных:")
print(f"   Обучение: {len(X_train)} примеров")
print(f"   Тестирование: {len(X_test)} примеров")

# Улучшенная модель
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAX_LEN, 39)),
    
    # Первый свёрточный блок
    tf.keras.layers.Conv1D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),
    
    # Второй свёрточный блок
    tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),
    
    # Третий свёрточный блок
    tf.keras.layers.Conv1D(128, 3, activation="relu", padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling1D(),
    
    # Полносвязные слои
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print("\n🧠 Архитектура модели:")
model.summary()

print("\n⏳ Начинаю обучение (это займёт 3-5 минут)...")

# Обучение с сохранением истории
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16,
    verbose=1
)

# Создаём папку models если её нет
os.makedirs("models", exist_ok=True)
model.save("models/stop_model.h5")
print("\n💾 Модель сохранена: models/stop_model.h5")

# Оценка модели
print("\n📊 ОЦЕНКА МОДЕЛИ НА ТЕСТОВЫХ ДАННЫХ:")
loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)

print(f"   Точность (Accuracy): {accuracy:.1%}")
print(f"   Precision: {precision:.1%} (мало ложных срабатываний)")
print(f"   Recall: {recall:.1%} (мало пропущенных 'стоп')")

# F1-score
f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
print(f"   F1-Score: {f1:.1%} (баланс точности и полноты)")

# Интерпретация результатов
print(f"\n🎯 ИНТЕРПРЕТАЦИЯ:")
if accuracy > 0.95:
    print("   ✅ ОТЛИЧНО! Модель готова к использованию.")
elif accuracy > 0.90:
    print("   👍 ХОРОШО! Можно использовать.")
elif accuracy > 0.85:
    print("   ⚠️ НОРМАЛЬНО! Можете добавить ещё данных.")
else:
    print("   ❌ ПЛОХО! Нужно больше разнообразных данных.")

# Графики обучения
try:
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Обучение')
    plt.plot(history.history['val_accuracy'], label='Валидация')
    plt.title('Точность модели')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Обучение')
    plt.plot(history.history['val_loss'], label='Валидация')
    plt.title('Потери модели')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=100)
    print(f"💾 Графики сохранены: models/training_history.png")
    
except:
    print("⚠️ Не удалось создать графики (возможно нет matplotlib)")

print("\n" + "="*60)
print("   🚀 МОДЕЛЬ ОБУЧЕНА!")
print("="*60)
print("Теперь запускайте браслет:")
print("   python app.py")
print("\nДля теста говорите 'стоп' с расстояния 1-1.5 метра.")