# vibration_controller.py
import json
import time


class VibrationController:
    def __init__(self):
        self.patterns = self.load_patterns()
        print("📳 Контроллер вибрации инициализирован")
    
    def load_patterns(self):
        """Загрузка паттернов вибрации из JSON"""
        try:
            with open("patterns.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ Файл patterns.json не найден, использую паттерны по умолчанию")
            return {
                "STOP_PATTERN": [200, 200, 200, 500],
                "HELP_PATTERN": [200, 100, 200, 100, 200, 500],
                "DANGER_PATTERN": [300, 100, 300, 100, 300],
            }
        except json.JSONDecodeError as e:
            print(f"⚠️ Ошибка чтения patterns.json: {e}")
            return {}
    
    def vibrate(self, pattern_name="STOP_PATTERN"):
        """
        Вибросигнал с визуализацией
        pattern_name: имя паттерна из patterns.json
        """
        if pattern_name not in self.patterns:
            print(f"⚠️ Паттерн '{pattern_name}' не найден. Доступные:")
            for key in self.patterns.keys():
                print(f"   - {key}")
            pattern_name = "STOP_PATTERN"  # используем по умолчанию
        
        pattern = self.patterns[pattern_name]
        
        print(f"\n📳 ВИБРАЦИЯ → {pattern_name}")
        print(f"   Паттерн: {pattern}")
        
        # Визуализация вибрации
        for i, duration in enumerate(pattern):
            if i % 2 == 0:  # вибрация
                bar = "█" * min(20, int(duration / 20))
                print(f"   ВИБРО [{bar:<20}] {duration}ms", end="\r")
            else:  # пауза
                print(f"   пауза {' ' * 19} {duration}ms", end="\r")
            
            # Эмуляция задержки
            time.sleep(duration / 1000)
        
        print()  # новая строка после завершения
    
    def vibrate_simple(self):
        """Простая вибрация без параметров (для обратной совместимости)"""
        self.vibrate("STOP_PATTERN")
    
    def test_all_patterns(self):
        """Тест всех паттернов вибрации"""
        print("\n🔧 ТЕСТ ВСЕХ ПАТТЕРНОВ ВИБРАЦИИ")
        for pattern_name in self.patterns:
            print(f"\nТестирую: {pattern_name}")
            self.vibrate(pattern_name)
            time.sleep(1)
        print("\n✅ Все паттерны протестированы")


# Для быстрого тестирования
if __name__ == "__main__":
    vibrator = VibrationController()
    print("Доступные паттерны:")
    for name, pattern in vibrator.patterns.items():
        print(f"  {name}: {pattern}")
    
    print("\nТест вибрации 'STOP'...")
    vibrator.vibrate("STOP_PATTERN")
    
    # Тест всех паттернов
    # vibrator.test_all_patterns()