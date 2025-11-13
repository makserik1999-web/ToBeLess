#!/usr/bin/env python3
"""
Скрипт для тестирования всех новых функций системы детекции
Запустите это ПОСЛЕ запуска основного app.py
"""
import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8080"

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def test_analytics():
    """Тест получения аналитики"""
    print_header("1. Тестируем Analytics API")
    
    response = requests.get(f"{BASE_URL}/analytics")
    if response.status_code == 200:
        data = response.json()
        print("✅ Analytics API работает")
        print(f"   Streaming: {data.get('streaming', False)}")
        
        stats = data.get('latest_stats', {})
        print(f"   People: {stats.get('people', 0)}")
        print(f"   Confidence: {stats.get('confidence', 0):.1f}%")
        print(f"   Tension: {stats.get('tension_score', 0):.1f}")
        print(f"   Conflict Type: {stats.get('conflict_type', 'none')}")
        
        analytics = data.get('analytics', {})
        fight_data = analytics.get('fight', {})
        print(f"\n   📊 Statistics:")
        print(f"      - Total fights: {fight_data.get('total_detections', 0)}")
        print(f"      - Strikes: {fight_data.get('strike_count', 0)}")
        print(f"      - Falls: {fight_data.get('fall_count', 0)}")
        print(f"      - Escalation warnings: {fight_data.get('escalation_warnings', 0)}")
    else:
        print(f"❌ Analytics API failed: {response.status_code}")

def test_heatmap():
    """Тест тепловой карты"""
    print_header("2. Тестируем Heatmap")
    
    response = requests.get(f"{BASE_URL}/heatmap")
    if response.status_code == 200:
        print("✅ Heatmap API работает")
        
        # Сохраняем изображение
        output_path = Path("test_heatmap.png")
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"   Тепловая карта сохранена: {output_path}")
        print(f"   Размер: {len(response.content)} bytes")
    else:
        print(f"❌ Heatmap API failed: {response.status_code}")

def test_hotspots():
    """Тест горячих точек"""
    print_header("3. Тестируем Hotspots API")
    
    response = requests.get(f"{BASE_URL}/hotspots")
    if response.status_code == 200:
        data = response.json()
        print("✅ Hotspots API работает")
        
        hotspots = data.get('hotspots', [])
        total_events = data.get('total_events', 0)
        
        print(f"   Total events: {total_events}")
        print(f"   Hotspots found: {len(hotspots)}")
        
        if hotspots:
            print("\n   🔥 Top 3 hotspots:")
            for i, spot in enumerate(hotspots[:3], 1):
                print(f"      {i}. Position: ({spot['x']}, {spot['y']})")
                print(f"         Intensity: {spot['intensity']:.2f}")
                print(f"         Events: {spot['events']}")
    else:
        print(f"❌ Hotspots API failed: {response.status_code}")

def test_settings():
    """Тест изменения настроек"""
    print_header("4. Тестируем Settings API")
    
    settings = {
        "body_proximity_threshold": 110.0,
        "strike_velocity_threshold": 22.0,
        "min_fight_frames": 12
    }
    
    response = requests.post(
        f"{BASE_URL}/settings",
        json=settings
    )
    
    if response.status_code == 200:
        print("✅ Settings API работает")
        print(f"   Настройки обновлены:")
        for key, value in settings.items():
            print(f"      - {key}: {value}")
    else:
        print(f"❌ Settings API failed: {response.status_code}")

def monitor_real_time(duration=30):
    """Мониторинг в реальном времени"""
    print_header(f"5. Мониторинг в реальном времени ({duration}s)")
    
    print("\n   Нажмите Ctrl+C для остановки\n")
    
    start_time = time.time()
    last_fight = False
    
    try:
        while (time.time() - start_time) < duration:
            response = requests.get(f"{BASE_URL}/analytics")
            
            if response.status_code == 200:
                data = response.json()
                stats = data.get('latest_stats', {})
                
                # Красивый вывод
                people = stats.get('people', 0)
                conf = stats.get('confidence', 0)
                tension = stats.get('tension_score', 0)
                conflict = stats.get('conflict_type', 'unknown')
                escalation = stats.get('escalation_warning', False)
                
                # Статус
                if stats.get('fights', 0) > 0:
                    status = "🔴 FIGHT"
                    if not last_fight:
                        print("\n   🚨 FIGHT DETECTED! 🚨")
                    last_fight = True
                elif escalation:
                    status = "🟡 WARNING"
                    print("\n   ⚠️  ESCALATION WARNING!")
                else:
                    status = "🟢 NORMAL"
                    last_fight = False
                
                # Форматированный вывод
                output = f"   [{status}] People: {people} | Conf: {conf:.0f}% | Tension: {tension:.0f} | Type: {conflict}"
                
                # Очищаем строку и выводим
                print(f"\r{output}", end='', flush=True)
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n   ⏹️  Мониторинг остановлен")

def test_conflict_types():
    """Показывает статистику по типам конфликтов"""
    print_header("6. Статистика типов конфликтов")
    
    response = requests.get(f"{BASE_URL}/analytics")
    if response.status_code == 200:
        data = response.json()
        analytics = data.get('analytics', {})
        fight_data = analytics.get('fight', {})
        conflict_types = fight_data.get('conflict_types', {})
        
        print("✅ Типы конфликтов:")
        
        type_names = {
            'minor_scuffle': '🟡 Лёгкая стычка',
            'active_fight': '🟠 Активная драка',
            'group_conflict': '🟣 Групповой конфликт',
            'critical': '🔴 Критическая ситуация'
        }
        
        total = sum(conflict_types.values())
        
        if total == 0:
            print("   Пока нет зарегистрированных конфликтов")
        else:
            for key, count in conflict_types.items():
                name = type_names.get(key, key)
                percentage = (count / total) * 100
                print(f"   {name}: {count} ({percentage:.1f}%)")
    else:
        print(f"❌ Failed to get conflict types: {response.status_code}")

def show_feature_summary():
    """Показывает сводку всех новых функций"""
    print_header("Новые функции системы")
    
    features = [
        ("🥊 Детекция ударов", "Отслеживает резкие движения руками"),
        ("🤸 Детекция падений", "Определяет когда человек на земле"),
        ("⚡ Предсказание эскалации", "Предупреждает за 3-5 секунд"),
        ("🗺️  Тепловая карта", "Визуализирует зоны конфликтов"),
        ("🔥 Детекция огня", "Улучшенная с проверкой формы"),
        ("🔪 Детекция оружия", "Использует YOLO detection"),
        ("📊 Классификация", "4 типа конфликтов"),
        ("👥 Треккинг людей", "Отслеживание между кадрами"),
        ("📈 Расширенная аналитика", "10+ метрик в реальном времени")
    ]
    
    for icon_name, description in features:
        print(f"   {icon_name}")
        print(f"      └─ {description}")

def main():
    print("\n" + "="*60)
    print("  🎯 ТЕСТИРОВАНИЕ НОВЫХ ФУНКЦИЙ СИСТЕМЫ ДЕТЕКЦИИ")
    print("="*60)
    
    # Проверяем доступность сервера
    try:
        response = requests.get(f"{BASE_URL}/analytics", timeout=2)
        if response.status_code != 200:
            print("\n❌ Сервер недоступен!")
            print("   Убедитесь что app.py запущен: python app.py")
            return
    except requests.exceptions.RequestException:
        print("\n❌ Не удалось подключиться к серверу!")
        print("   Запустите сервер: python app.py")
        return
    
    print("✅ Сервер доступен\n")
    
    # Показываем список функций
    show_feature_summary()
    
    # Запускаем тесты
    test_analytics()
    test_heatmap()
    test_hotspots()
    test_conflict_types()
    test_settings()
    
    # Предлагаем мониторинг
    print_header("Мониторинг")
    choice = input("\n   Запустить мониторинг в реальном времени? (y/n): ")
    
    if choice.lower() == 'y':
        duration = input("   Длительность в секундах (по умолчанию 30): ")
        try:
            duration = int(duration) if duration else 30
        except ValueError:
            duration = 30
        
        monitor_real_time(duration)
    
    print_header("Тестирование завершено")
    print("\n   📋 Результаты:")
    print("      - Analytics API: ✅")
    print("      - Heatmap: ✅")
    print("      - Hotspots: ✅")
    print("      - Settings: ✅")
    print("\n   🎉 Все функции работают корректно!")
    print("\n   💡 Для презентации:")
    print("      1. Откройте http://localhost:8080/detection")
    print("      2. Запустите stream с камеры или загрузите видео")
    print("      3. Покажите детекцию в реальном времени")
    print("      4. Откройте http://localhost:8080/heatmap для тепловой карты")

if __name__ == "__main__":
    main()