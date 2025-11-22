#!/usr/bin/env python3
import os
import sys
import time
import argparse
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import select
import binascii
import socket
import struct
import json
import hashlib
from datetime import datetime
import threading
from queue import Queue
import platform
import shutil
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

# Конфигурация
CSI_PORT = 5500
HEADER_FORMAT = 'HHHHHHI'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MOTION_THRESHOLD = 5.0
INTERFERENCE_THRESHOLD = 15
CALIBRATION_TIME = 5.0
SCAN_INTERVAL = 0.5
ADAPTIVE_WINDOW = 60.0
STABILITY_THRESHOLD = 1.5
ADAPTATION_RATE = 0.2
DYNAMIC_THRESHOLD_FACTOR = 2.0
NEAR_ROUTER_THRESHOLD = 0.7
WHOFI_PROFILES_FILE = "who_fi_profiles.json"
WHOFI_MIN_FEATURES = 50
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
LOG_FILE = "wifi_sensing.log"
ROOM_MODEL_FILE = "room_model.json"

# Глобальные состояния
WHOFI_TRAINING_MODE = False
WHOFI_CURRENT_USER = ""
WHOFI_ALERT_UNKNOWN = True
ADAPTER_COMPATIBILITY = {}
CURRENT_CHANNEL = 0
CURRENT_MODE = ""
STOP_EVENT = threading.Event()

class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clear_screen():
    """Очистка экрана терминала"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    """Печать заголовка"""
    clear_screen()
    print(f"{TerminalColors.HEADER}{'=' * 60}")
    print(f"{title.center(60)}")
    print(f"{'=' * 60}{TerminalColors.ENDC}\n")

def print_menu(options):
    """Отображение меню с опциями"""
    for i, option in enumerate(options, 1):
        print(f"{TerminalColors.BOLD}{i}.{TerminalColors.ENDC} {option}")
    print()

def safe_write_log(message):
    """Безопасная запись в лог с контролем размера"""
    try:
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > MAX_LOG_SIZE:
            backup = f"{LOG_FILE}.bak"
            if os.path.exists(backup):
                os.remove(backup)
            os.rename(LOG_FILE, backup)
        
        with open(LOG_FILE, "a") as log:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"Ошибка записи лога: {str(e)}")

def get_wireless_interfaces():
    """Получает список беспроводных интерфейсов с проверкой поддержки монитора"""
    interfaces = []
    for root, dirs, files in os.walk('/sys/class/net'):
        for name in dirs:
            iface_path = os.path.join(root, name)
            if 'wireless' in os.listdir(iface_path):
                # Проверка поддержки режима монитора
                supported_modes = []
                try:
                    with open(os.path.join(iface_path, 'modes'), 'r') as f:
                        supported_modes = f.read().strip().split()
                except:
                    pass
                
                # Проверка CSI поддержки (эвристика по известным адаптерам)
                csi_support = False
                known_csi_adapters = ['iwlwifi', 'ath9k', 'ath10k', 'rt2800usb']
                driver_path = os.path.join(iface_path, 'device', 'driver', 'module')
                if os.path.exists(driver_path):
                    try:
                        driver_name = os.path.basename(os.readlink(driver_path))
                        csi_support = any(adapter in driver_name for adapter in known_csi_adapters)
                    except:
                        pass
                
                interfaces.append({
                    'name': name,
                    'monitor_support': 'monitor' in supported_modes,
                    'csi_support': csi_support
                })
                
                # Кэширование информации
                ADAPTER_COMPATIBILITY[name] = {
                    'monitor': 'monitor' in supported_modes,
                    'csi': csi_support
                }
    
    return interfaces

def select_interface():
    """Интерактивный выбор адаптера"""
    interfaces = get_wireless_interfaces()
    
    if not interfaces:
        print(f"{TerminalColors.FAIL}Беспроводные интерфейсы не найдены!{TerminalColors.ENDC}")
        exit(1)
    
    print_header("ВЫБОР БЕСПРОВОДНОГО АДАПТЕРА")
    
    print(f"{TerminalColors.BOLD}Доступные адаптеры:{TerminalColors.ENDC}")
    print(f"{'№':<3} {'Имя':<10} {'Монитор':<10} {'CSI':<10} {'Поддержка'}")
    print("-" * 40)
    
    for i, iface in enumerate(interfaces, 1):
        monitor_sym = f"{TerminalColors.OKGREEN}✓{TerminalColors.ENDC}" if iface['monitor_support'] else f"{TerminalColors.FAIL}✗{TerminalColors.ENDC}"
        csi_sym = f"{TerminalColors.OKGREEN}✓{TerminalColors.ENDC}" if iface['csi_support'] else f"{TerminalColors.FAIL}✗{TerminalColors.ENDC}"
        print(f"{i:<3} {iface['name']:<10} {monitor_sym:<10} {csi_sym:<10}")

    print("\n0. Выход")
    
    while True:
        try:
            choice = int(input("\nВыберите адаптер: "))
            if choice == 0:
                exit(0)
            if 1 <= choice <= len(interfaces):
                return interfaces[choice-1]['name']
            print("Неверный выбор. Попробуйте снова.")
        except ValueError:
            print("Пожалуйста, введите число.")

def scan_networks(interface):
    """Сканирует Wi-Fi сети и возвращает список с деталями"""
    print(f"\nСканирование сетей на {interface}...")
    
    try:
        # Для разных систем используем разные команды
        if platform.system() == "Linux":
            cmd = ['sudo', 'iw', 'dev', interface, 'scan']
        elif platform.system() == "Darwin":  # macOS
            cmd = ['/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport', '-s']
        else:
            print(f"{TerminalColors.FAIL}Неподдерживаемая ОС{TerminalColors.ENDC}")
            return []
        
        result = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            timeout=15
        ).decode('utf-8', errors='ignore')
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Ошибка сканирования: {str(e)}")
        return []
    except FileNotFoundError:
        print("Команда сканирования не найдена. Установите необходимые утилиты.")
        return []

    networks = []
    current_net = {}
    
    # Разные парсеры для разных систем
    if platform.system() == "Linux":
        for line in result.split('\n'):
            if 'BSS' in line:
                if current_net:
                    networks.append(current_net)
                bssid = re.search(r'([0-9a-fA-F:]{17})', line)
                current_net = {'bssid': bssid.group(1) if bssid else 'unknown'}
            
            elif 'SSID:' in line:
                current_net['ssid'] = line.split('SSID:')[-1].strip()
            
            elif 'signal:' in line:
                signal = re.search(r'(-?\d+\.\d+) dBm', line)
                if signal:
                    current_net['rssi'] = float(signal.group(1))
            
            elif 'freq:' in line:
                freq = re.search(r'(\d+)', line)
                if freq:
                    current_net['freq'] = int(freq.group(1))
                    current_net['channel'] = freq_to_channel(int(freq.group(1)))
    elif platform.system() == "Darwin":
        # Парсинг вывода airport на macOS
        lines = result.split('\n')[1:]  # Пропускаем заголовок
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                bssid = parts[1]
                ssid = ' '.join(parts[2:-2])
                rssi = int(parts[-2])
                channel = parts[-1].split(',')[0]
                networks.append({
                    'bssid': bssid,
                    'ssid': ssid,
                    'rssi': rssi,
                    'channel': int(channel) if channel.isdigit() else 0
                })
    
    if current_net:
        networks.append(current_net)
    
    return networks

def freq_to_channel(freq):
    """Преобразование частоты в номер канала"""
    if 2400 <= freq <= 2500:
        return (freq - 2407) // 5
    elif 5000 <= freq <= 6000:
        return (freq - 5000) // 5
    return 0

def select_network(interface):
    """Интерактивный выбор сети"""
    networks = scan_networks(interface)
    
    if not networks:
        print(f"{TerminalColors.FAIL}Сети не найдены. Убедитесь, что роутеры включены.{TerminalColors.ENDC}")
        return None, 0
    
    print_header("ВЫБОР СЕТИ ДЛЯ МОНИТОРИНГА")
    
    print(f"{'№':<3} {'BSSID':<18} {'RSSI':<6} {'Канал':<6} {'SSID'}")
    print("-" * 60)
    
    visible = []
    for i, net in enumerate(networks, 1):
        bssid = net.get('bssid', 'unknown')
        rssi = net.get('rssi', 0)
        channel = net.get('channel', 0)
        ssid = (net.get('ssid', 'hidden') or 'hidden')[:20]
        
        if bssid == 'unknown' or rssi == 0:
            continue
            
        rssi_color = TerminalColors.FAIL if rssi < -80 else TerminalColors.WARNING if rssi < -70 else TerminalColors.OKGREEN
        print(f"{i:<3} {bssid:<18} {rssi_color}{rssi:>5}{TerminalColors.ENDC} dBm {channel:^6} {ssid}")
        visible.append(net)
    
    if not visible:
        return None, 0
    
    print("\n0. Назад")
    
    while True:
        try:
            choice = int(input("\nВыберите сеть: "))
            if choice == 0:
                return None, 0
            if 1 <= choice <= len(visible):
                return visible[choice-1]['bssid'], visible[choice-1].get('channel', 6)
            print("Неверный выбор.")
        except ValueError:
            print("Введите число.")

def setup_interface(interface, mode="managed"):
    """Настройка интерфейса в нужном режиме"""
    try:
        # Проверка текущего режима
        current_mode = ""
        try:
            result = subprocess.check_output(
                ['iw', 'dev', interface, 'info'],
                stderr=subprocess.STDOUT
            ).decode('utf-8')
            if 'type monitor' in result:
                current_mode = "monitor"
            elif 'type managed' in result:
                current_mode = "managed"
        except:
            pass
        
        if current_mode == mode:
            return True
            
        # Изменение режима
        os.system(f'sudo ip link set {interface} down')
        os.system(f'sudo iw dev {interface} set type {mode}')
        os.system(f'sudo ip link set {interface} up')
        
        # Проверка успешности
        time.sleep(1)
        result = subprocess.check_output(
            ['iw', 'dev', interface, 'info'],
            stderr=subprocess.STDOUT
        ).decode('utf-8')
        
        if f'type {mode}' in result:
            print(f"Интерфейс {interface} переведен в режим {mode}")
            return True
        else:
            print(f"{TerminalColors.FAIL}Не удалось перевести интерфейс в режим {mode}{TerminalColors.ENDC}")
            return False
    except Exception as e:
        print(f"Ошибка настройки интерфейса: {str(e)}")
        return False

def setup_csi_socket(interface, target_mac, channel=6):
    """Настройка сокета для CSI режима"""
    try:
        # Установка канала
        os.system(f'sudo iw dev {interface} set channel {channel} HT20')
        time.sleep(1)
        
        sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0003))
        sock.bind((interface, 0))
        
        # Фильтр по MAC
        filter_prog = [
            [0x28, 0, 0, 0x0000000c],
            [0x15, 0, 3, int.from_bytes(target_mac[:4], 'big')],
            [0x15, 0, 2, int.from_bytes(target_mac[4:], 'big')],
            [0x6, 0, 0, 0x0000ffff],
            [0x6, 0, 0, 0],
        ]
        
        sock.setsockopt(socket.SOL_SOCKET, 0x26, struct.pack('HL', 5, *filter_prog))
        return sock
    except Exception as e:
        print(f"Ошибка создания сокета: {str(e)}")
        return None

def parse_csi_packet(data):
    """Разбор CSI пакета с улучшенной обработкой ошибок"""
    try:
        if len(data) < HEADER_SIZE + 128:
            return None, None
        
        header = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
        csi_data = np.frombuffer(data[HEADER_SIZE:HEADER_SIZE+128], dtype=np.int8)
        
        # Проверка на нулевые значения
        if np.all(csi_data == 0):
            return None, None
            
        csi_complex = csi_data[::2] + 1j * csi_data[1::2]
        return np.abs(csi_complex), header
    except Exception as e:
        safe_write_log(f"Ошибка разбора пакета: {str(e)}")
        return None, None

def detect_motion_direction(current, baseline):
    """Определение направления движения (CSI)"""
    left_side = current[:21] - baseline[:21]
    center_side = current[21:43] - baseline[21:43]
    right_side = current[43:] - baseline[43:]
    
    left_diff = np.mean(left_side)
    center_diff = np.mean(center_side)
    right_diff = np.mean(right_side)
    
    max_diff = max(left_diff, center_diff, right_diff, key=abs)
    
    if abs(max_diff) < MOTION_THRESHOLD:
        return "none", 0
    
    if max_diff == left_diff:
        return "left", left_diff
    elif max_diff == right_diff:
        return "right", right_diff
    else:
        return "center", center_diff

def detect_interference(amplitudes):
    """Обнаружение помех (CSI)"""
    high_freq = amplitudes[52:60]
    variance = np.var(high_freq)
    
    if variance > INTERFERENCE_THRESHOLD:
        if any(amp > 35 for amp in high_freq):
            return "impulse", variance
        elif np.mean(high_freq) > 30:
            return "continuous", variance
        else:
            return "fluctuating", variance
    return "none", variance

def detect_near_router(amplitudes, baseline):
    """Определение движения рядом с роутером"""
    high_freq = amplitudes[52:60] - baseline[52:60]
    low_freq = amplitudes[0:10] - baseline[0:10]
    
    ratio = np.mean(np.abs(high_freq)) / (np.mean(np.abs(low_freq)) + 0.001)
    
    return ratio > NEAR_ROUTER_THRESHOLD, ratio

def get_rssi(interface):
    """Получение RSSI для стандартного режима"""
    try:
        if platform.system() == "Linux":
            result = subprocess.check_output(
                ['iwconfig', interface],
                stderr=subprocess.STDOUT
            ).decode('utf-8')
            
            match = re.search(r'Signal level=(-?\d+) dBm', result)
            if match:
                return int(match.group(1))
        elif platform.system() == "Darwin":
            result = subprocess.check_output(
                ['/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport', '-I'],
                stderr=subprocess.STDOUT
            ).decode('utf-8')
            
            match = re.search(r'agrCtlRSSI:\s*(-?\d+)', result)
            if match:
                return int(match.group(1))
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None

def adaptive_calibration(values, baseline, last_calib_time, current_time, window_size=20):
    """Адаптивная система калибровки базового уровня"""
    if not values:
        return baseline, last_calib_time
    
    recent_values = list(values)[-window_size:]
    mean_val = np.mean(recent_values)
    std_val = np.std(recent_values)
    
    refresh_conditions = [
        std_val < STABILITY_THRESHOLD,
        current_time - last_calib_time > ADAPTIVE_WINDOW,
        abs(mean_val - baseline) > 3 * STABILITY_THRESHOLD
    ]
    
    if any(refresh_conditions):
        alpha = 0.2 if std_val < STABILITY_THRESHOLD else 0.05
        new_baseline = alpha * mean_val + (1 - alpha) * baseline
        return new_baseline, current_time
    
    return baseline, last_calib_time

# ================= Who-Fi Функциональность =================

def load_who_fi_profiles():
    """Загрузка профилей Who-Fi из файла"""
    if os.path.exists(WHOFI_PROFILES_FILE):
        try:
            with open(WHOFI_PROFILES_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_who_fi_profiles(profiles):
    """Сохранение профилей Who-Fi в файл"""
    try:
        with open(WHOFI_PROFILES_FILE, 'w') as f:
            json.dump(profiles, f, indent=2)
    except Exception as e:
        safe_write_log(f"Ошибка сохранения профилей Who-Fi: {str(e)}")

def extract_who_fi_features(amplitudes):
    """Извлечение признаков для Who-Fi идентификации"""
    # Основные статистические характеристики
    features = [
        np.mean(amplitudes),
        np.std(amplitudes),
        np.min(amplitudes),
        np.max(amplitudes),
        np.median(amplitudes),
        np.percentile(amplitudes, 25),
        np.percentile(amplitudes, 75)
    ]
    
    # Спектральные характеристики
    fft = np.fft.fft(amplitudes)
    features.append(np.abs(fft[1]))  # Первая гармоника
    features.append(np.abs(fft[2]))  # Вторая гармоника
    
    # Отношение высокочастотных/низкочастотных компонент
    hf = np.mean(amplitudes[48:])
    lf = np.mean(amplitudes[:16])
    features.append(hf / (lf + 0.001))
    
    # Энтропия
    hist, _ = np.histogram(amplitudes, bins=10)
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    features.append(entropy)
    
    return features

def who_fi_identify(features, profiles):
    """Идентификация человека с использованием ML"""
    if not profiles or len(profiles) < 2:
        return None, 0.0
    
    # Подготовка данных для классификации
    X = []
    y = []
    for user_id, profile in profiles.items():
        if 'training_data' in profile:
            for sample in profile['training_data']:
                X.append(sample)
                y.append(user_id)
    
    if len(set(y)) < 2:
        return None, 0.0
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Создание и обучение модели
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=0.95),
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    model.fit(X_train, y_train)
    
    # Проверка точности
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    safe_write_log(f"[Who-Fi] Точность модели: {accuracy:.2f}")
    
    # Предсказание для текущего образца
    user_id = model.predict([features])[0]
    proba = np.max(model.predict_proba([features]))
    
    return user_id, proba

def who_fi_activity_recognition(sequence):
    """Распознавание типа активности"""
    if len(sequence) < 5:
        return "unknown", 0.5
    
    # Анализ временных характеристик
    diff_seq = np.abs(np.diff(sequence))
    mean_change = np.mean(diff_seq)
    max_change = np.max(diff_seq)
    var_change = np.var(diff_seq)
    
    # Частотный анализ
    fft = np.abs(np.fft.rfft(diff_seq))
    dominant_freq = np.argmax(fft[1:]) + 1
    
    # Логика классификации
    if mean_change > 5.0 and max_change > 15.0 and dominant_freq > 3:
        return "running", min(0.9, mean_change/10)
    elif mean_change > 2.0 and 1 <= dominant_freq <= 3:
        return "walking", min(0.8, mean_change/5)
    elif mean_change < 1.0 and var_change < 0.5:
        return "sitting/standing", 0.7
    elif mean_change < 0.5 and var_change < 0.1:
        return "no movement", 0.9
    else:
        return "unknown", 0.5

def who_fi_alert(message):
    """Система оповещений Who-Fi"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n{TerminalColors.FAIL}[Who-Fi ALERT] [{timestamp}] {message}{TerminalColors.ENDC}")
    safe_write_log(f"[ALERT] {message}")
    
    # Здесь можно добавить отправку уведомления:
    # - Email
    # - Telegram
    # - Звуковой сигнал

def who_fi_train_new_user(interface, target_mac, user_name, channel=6):
    """Режим обучения для нового пользователя"""
    print(f"\n{TerminalColors.OKBLUE}[Who-Fi TRAINING] Начато обучение для: {user_name}{TerminalColors.ENDC}")
    print("Пожалуйста, двигайтесь в зоне действия в течение 30 секунд...")
    
    # Настройка интерфейса
    if not setup_interface(interface, "monitor"):
        return
    
    # Настройка сокета
    target_mac_bin = binascii.unhexlify(target_mac.replace(':', ''))
    sock = setup_csi_socket(interface, target_mac_bin, channel)
    if sock is None:
        return
    
    features = []
    start_time = time.time()
    
    try:
        while time.time() - start_time < 30 and not STOP_EVENT.is_set():
            data = sock.recv(2048)
            amplitudes, _ = parse_csi_packet(data)
            if amplitudes is None:
                continue
                
            # Извлечение признаков
            feat = extract_who_fi_features(amplitudes)
            features.append(feat)
            
            # Прогресс
            elapsed = time.time() - start_time
            print(f"\rПрогресс: {int(elapsed)}/30 сек | Образцов: {len(features)}", end='')
            sys.stdout.flush()
            
    except KeyboardInterrupt:
        print("\nОбучение прервано")
        return
    except Exception as e:
        print(f"\nОшибка при обучении: {str(e)}")
        return
    finally:
        sock.close()
        setup_interface(interface, "managed")
    
    if len(features) < 20:
        print("\nНедостаточно данных для обучения. Попробуйте еще раз.")
        return
    
    # Вычисление среднего профиля
    avg_features = np.mean(features, axis=0).tolist()
    
    # Создание ID пользователя
    user_id = hashlib.sha256(user_name.encode()).hexdigest()[:8]
    
    # Загрузка существующих профилей
    profiles = load_who_fi_profiles()
    
    # Добавление нового профиля
    profiles[user_id] = {
        "name": user_name,
        "features": avg_features,
        "training_data": features,  # Все собранные образцы
        "created": datetime.now().isoformat(),
        "samples": len(features)
    }
    
    # Сохранение профилей
    save_who_fi_profiles(profiles)
    print(f"\n{TerminalColors.OKGREEN}[Who-Fi] Профиль '{user_name}' успешно создан! ID: {user_id}{TerminalColors.ENDC}")

def csi_mode(interface, target_mac, channel=6):
    """Режим работы с CSI данными"""
    global WHOFI_TRAINING_MODE, WHOFI_CURRENT_USER, CURRENT_MODE
    
    # Загрузка Who-Fi профилей
    who_fi_profiles = load_who_fi_profiles()
    
    # Режим обучения нового пользователя
    if WHOFI_TRAINING_MODE and WHOFI_CURRENT_USER:
        who_fi_train_new_user(interface, target_mac, WHOFI_CURRENT_USER, channel)
        return
    
    # Настройка интерфейса
    if not setup_interface(interface, "monitor"):
        return
    
    # Настройка сокета
    target_mac_bin = binascii.unhexlify(target_mac.replace(':', ''))
    sock = setup_csi_socket(interface, target_mac_bin, channel)
    if sock is None:
        return
    
    print(f"\n{TerminalColors.OKBLUE}CSI режим: отслеживание {target_mac}{TerminalColors.ENDC}")
    if who_fi_profiles:
        print(f"[Who-Fi] Загружено профилей: {len(who_fi_profiles)}")
    else:
        print("[Who-Fi] Профили не найдены. Идентификация отключена.")
    
    # Первоначальная калибровка
    calibration_data = []
    start_calib = time.time()
    while time.time() - start_calib < CALIBRATION_TIME and not STOP_EVENT.is_set():
        rlist, _, _ = select.select([sock], [], [], 0.1)
        if rlist:
            data = sock.recv(2048)
            amps, _ = parse_csi_packet(data)
            if amps is not None:
                calibration_data.append(amps)
    
    if not calibration_data:
        print("Ошибка калибровки. Проверьте сигнал.")
        sock.close()
        setup_interface(interface, "managed")
        return
    
    baseline = np.mean(calibration_data, axis=0)
    print("Первоначальная калибровка завершена. Начинаем мониторинг...")
    
    # Настройка графиков
    plt.ion()
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2)
    
    # График CSI
    ax1 = fig.add_subplot(gs[0, :])
    sc = np.arange(-32, 32)
    line_csi, = ax1.plot(sc, np.zeros(64), 'b-', label='Текущий')
    line_base, = ax1.plot(sc, baseline, 'g--', label='Базовый')
    ax1.set_title(f'CSI: {target_mac}')
    ax1.set_xlabel('Поднесущая')
    ax1.set_ylabel('Амплитуда')
    ax1.set_ylim(0, 50)
    ax1.grid(True)
    ax1.legend()
    
    # График RSSI
    ax2 = fig.add_subplot(gs[1, 0])
    rssi_vals = deque(maxlen=100)
    time_vals = deque(maxlen=100)
    line_rssi, = ax2.plot([], [], 'r-')
    ax2.set_title('История RSSI')
    ax2.set_xlabel('Время (сек)')
    ax2.set_ylabel('RSSI (dBm)')
    ax2.set_ylim(-100, -20)
    ax2.grid(True)
    
    # График разницы
    ax3 = fig.add_subplot(gs[1, 1])
    diff_vals = deque(maxlen=100)
    line_diff, = ax3.plot([], [], 'm-')
    ax3.set_title('Разница CSI')
    ax3.set_xlabel('Время (сек)')
    ax3.set_ylabel('Средняя разница')
    ax3.grid(True)
    ax3.axhline(y=MOTION_THRESHOLD, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=-MOTION_THRESHOLD, color='r', linestyle='--', alpha=0.5)
    
    # Статусы
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    status = ax4.text(0.02, 0.8, '', fontsize=12)
    motion = ax4.text(0.02, 0.6, 'Движение: нет', color='green', fontsize=12)
    direction = ax4.text(0.02, 0.4, 'Направление: -', fontsize=12)
    interf = ax4.text(0.02, 0.2, 'Помехи: нет', color='green', fontsize=12)
    who_fi_status = ax4.text(0.02, 0.0, 'Who-Fi: Ожидание данных...', color='blue', fontsize=12)
    
    plt.tight_layout()
    
    # Who-Fi переменные
    motion_sequence = []
    last_identified_user = None
    last_activity_time = 0
    unknown_detected = False
    
    start_time = time.time()
    last_update = start_time
    last_motion = 0
    last_calib = start_time
    motion_cooldown = 2.0
    recent_amps = deque(maxlen=20)
    
    CURRENT_MODE = "CSI"
    
    try:
        while not STOP_EVENT.is_set():
            rlist, _, _ = select.select([sock], [], [], 0.1)
            if rlist:
                data = sock.recv(2048)
                amps, header = parse_csi_packet(data)
                if amps is None:
                    continue
                
                recent_amps.append(amps)
                rssi = header[3]
                now = time.time()
                elapsed = now - start_time
                
                time_vals.append(elapsed)
                rssi_vals.append(rssi)
                diff = np.mean(amps - baseline)
                diff_vals.append(diff)
                
                # Адаптивная калибровка
                if now - last_calib > ADAPTATION_RATE:
                    baseline, last_calib = adaptive_calibration(
                        recent_amps, baseline, last_calib, now
                    )
                    line_base.set_ydata(baseline)
                
                # Детекция движения
                dir, diff_val = detect_motion_direction(amps, baseline)
                if dir != "none":
                    last_motion = now
                    motion.set_text('Движение: ОБНАРУЖЕНО!')
                    motion.set_color('red')
                    direction.set_text(f'Направление: {dir.capitalize()} (Δ={diff_val:.1f})')
                    
                    # Определение близости к роутеру
                    near, ratio = detect_near_router(amps, baseline)
                    status_text = "да" if near else "нет"
                    color = "red" if near else "blue"
                    
                    # Who-Fi: Сбор данных для идентификации
                    if len(motion_sequence) < 20:
                        motion_sequence.append(amps)
                elif now - last_motion > motion_cooldown:
                    motion.set_text('Движение: нет')
                    motion.set_color('green')
                    direction.set_text('Направление: -')
                    
                    # Who-Fi: Анализ собранной последовательности
                    if motion_sequence and who_fi_profiles:
                        # Извлечение признаков из всей последовательности
                        sequence_features = []
                        for amp in motion_sequence:
                            sequence_features.extend(extract_who_fi_features(amp))
                        
                        # Идентификация пользователя
                        user_id, confidence = who_fi_identify(
                            sequence_features, 
                            who_fi_profiles
                        )
                        
                        if user_id and confidence > 0.6:
                            user_name = who_fi_profiles[user_id]["name"]
                            who_fi_status.set_text(
                                f'Who-Fi: {user_name} (уверенность: {confidence:.2f})'
                            )
                            who_fi_status.set_color('green')
                            last_identified_user = user_name
                        else:
                            who_fi_status.set_text('Who-Fi: Неизвестный человек!')
                            who_fi_status.set_color('red')
                            unknown_detected = True
                            
                            # Оповещение о незнакомце
                            if WHOFI_ALERT_UNKNOWN and now - last_activity_time > 30:
                                who_fi_alert("Обнаружен неизвестный человек!")
                                last_activity_time = now
                        
                        # Распознавание активности
                        activity, activity_conf = who_fi_activity_recognition(
                            [np.mean(a) for a in motion_sequence]
                        )
                        safe_write_log(f"[Who-Fi] Активность: {activity} ({activity_conf:.2f})")
                        
                        motion_sequence = []
                    
                # Детекция помех
                intf, var_val = detect_interference(amps)
                if intf != "none":
                    interf.set_text(f'Помехи: {intf} (var={var_val:.1f})')
                    interf.set_color('red')
                else:
                    interf.set_text('Помехи: нет')
                    interf.set_color('green')
                
                # Обновление графиков
                line_csi.set_ydata(amps)
                line_rssi.set_data(time_vals, rssi_vals)
                line_diff.set_data(time_vals, diff_vals)
                
                if time_vals:
                    ax2.set_xlim(max(0, time_vals[0]), time_vals[-1])
                    ax3.set_xlim(max(0, time_vals[0]), time_vals[-1])
                    if diff_vals:
                        ax3.set_ylim(min(diff_vals)-1, max(diff_vals)+1)
                
                status.set_text(
                    f'Время: {elapsed:.1f} сек | RSSI: {rssi} dBm | '
                    f'Пакетов: {len(time_vals)} | Движение: {now - last_motion:.1f} сек назад'
                )
                
                if now - last_update > 0.3:
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    last_update = now
                    
    except KeyboardInterrupt:
        print("\nМониторинг остановлен")
    except Exception as e:
        print(f"\nОшибка в CSI режиме: {str(e)}")
        traceback.print_exc()
    finally:
        sock.close()
        setup_interface(interface, "managed")
        plt.ioff()
        if not STOP_EVENT.is_set():
            plt.show()

def rssi_mode(interface, target_mac):
    """Стандартный режим работы с RSSI"""
    global CURRENT_MODE
    
    print(f"\n{TerminalColors.OKBLUE}RSSI режим: отслеживание {target_mac}{TerminalColors.ENDC}")
    print("Для остановки нажмите Ctrl+C...")
    
    # Настройка интерфейса
    if not setup_interface(interface, "managed"):
        return
    
    # Настройка графиков
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f'RSSI мониторинг: {target_mac}')
    ax.set_xlabel('Время (сек)')
    ax.set_ylabel('RSSI (dBm)')
    ax.grid(True)
    
    rssi_vals = deque(maxlen=100)
    time_vals = deque(maxlen=100)
    line, = ax.plot([], [], 'b-')
    ax.set_ylim(-100, -20)
    
    start_time = time.time()
    last_update = start_time
    last_calib = start_time
    baseline = 0
    baseline_history = deque(maxlen=50)
    std_history = deque(maxlen=20)
    baseline_calibrated = False
    last_rssi = None
    rssi_change_history = deque(maxlen=5)
    
    CURRENT_MODE = "RSSI"
    
    try:
        while not STOP_EVENT.is_set():
            rssi = get_rssi(interface)
            current_time = time.time()
            elapsed = current_time - start_time
            
            if rssi is not None:
                # Калибровка базового уровня
                if not baseline_calibrated and elapsed > CALIBRATION_TIME:
                    baseline = sum(rssi_vals) / len(rssi_vals) if rssi_vals else rssi
                    print(f"Первоначальная калибровка завершена. Базовый RSSI: {baseline:.1f} dBm")
                    baseline_calibrated = True
                
                # Сохранение данных
                time_vals.append(elapsed)
                rssi_vals.append(rssi)
                
                # Адаптивная калибровка
                if baseline_calibrated and current_time - last_calib > ADAPTATION_RATE:
                    baseline, last_calib = adaptive_calibration(
                        rssi_vals, baseline, last_calib, current_time
                    )
                    baseline_history.append(baseline)
                    
                    recent_rssi = list(rssi_vals)[-20:]
                    if recent_rssi:
                        std_val = np.std(recent_rssi)
                        std_history.append(std_val)
                
                # Определение изменения сигнала
                if last_rssi is not None:
                    rssi_change_history.append(rssi - last_rssi)
                last_rssi = rssi
                
                # Обновление графика
                line.set_data(time_vals, rssi_vals)
                
                # Визуализация
                if baseline_history and std_history:
                    ax.plot(time_vals, list(baseline_history)[-len(time_vals):], 'g--', alpha=0.7, label='Адапт. базовый')
                    if len(time_vals) > 1:
                        ax.fill_between(time_vals, 
                                        [b - 2*s for b, s in zip(list(baseline_history)[-len(time_vals):], list(std_history)[-len(time_vals):])],
                                        [b + 2*s for b, s in zip(list(baseline_history)[-len(time_vals):], list(std_history)[-len(time_vals):])],
                                        color='gray', alpha=0.1)
                
                if time_vals:
                    ax.set_xlim(max(0, time_vals[0]), time_vals[-1])
                
                # Детекция движения и оценка положения
                dynamic_threshold = DYNAMIC_THRESHOLD_FACTOR * (np.mean(std_history) if std_history else MOTION_THRESHOLD)
                diff = rssi - baseline
                
                if baseline_calibrated:
                    if abs(diff) > dynamic_threshold:
                        color = 'red'
                        
                        # Эвристика для определения "близости к роутеру"
                        near_router = False
                        if rssi_change_history:
                            avg_change = np.mean(np.abs(list(rssi_change_history)))
                            near_router = avg_change > 2.0
                        
                        status_text = f"Обнаружено движение! Отклонение: {diff:.1f} dB"
                        if near_router:
                            status_text += " | Возможно рядом с роутером"
                    else:
                        color = 'blue'
                        status_text = f"Стабильный сигнал. Отклонение: {diff:.1f} dB"
                    
                    line.set_color(color)
                    ax.set_title(f'RSSI: {target_mac} | {status_text}')
            
            # Обновление экрана
            if current_time - last_update > 0.3:
                fig.canvas.draw()
                fig.canvas.flush_events()
                last_update = current_time
            
            time.sleep(SCAN_INTERVAL)
            
    except KeyboardInterrupt:
        print("\nМониторинг остановлен")
    except Exception as e:
        print(f"\nОшибка в RSSI режиме: {str(e)}")
    finally:
        plt.ioff()
        if not STOP_EVENT.is_set():
            plt.show()

def who_fi_menu(interface, target_mac, channel):
    """Меню для работы с Who-Fi"""
    global WHOFI_TRAINING_MODE, WHOFI_CURRENT_USER
    
    profiles = load_who_fi_profiles()
    
    while True:
        print_header("СИСТЕМА WHO-FI")
        
        if profiles:
            print(f"{TerminalColors.BOLD}Зарегистрированные пользователи:{TerminalColors.ENDC}")
            print("-" * 50)
            for user_id, profile in profiles.items():
                print(f"• {profile['name']} ({user_id}) - {profile['created'][:10]}")
            print("-" * 50)
        else:
            print("Нет зарегистрированных пользователей")
        
        options = [
            "Обучить нового пользователя",
            "Удалить профиль пользователя",
            "Проверить идентификацию в реальном времени",
            "Назад"
        ]
        print_menu(options)
        
        choice = input("Выберите действие: ")
        
        if choice == "1":
            user_name = input("\nВведите имя пользователя: ")
            if user_name:
                WHOFI_TRAINING_MODE = True
                WHOFI_CURRENT_USER = user_name
                csi_mode(interface, target_mac, channel)
                WHOFI_TRAINING_MODE = False
                input("\nНажмите Enter для продолжения...")
        
        elif choice == "2":
            if not profiles:
                input("Нет профилей для удаления. Нажмите Enter...")
                continue
                
            user_id = input("\nВведите ID пользователя для удаления: ")
            if user_id in profiles:
                del profiles[user_id]
                save_who_fi_profiles(profiles)
                print(f"Профиль {user_id} удален")
            else:
                print("Профиль не найден")
            input("\nНажмите Enter для продолжения...")
        
        elif choice == "3":
            if not profiles:
                input("Нет профилей для идентификации. Нажмите Enter...")
                continue
                
            print("\nЗапуск идентификации в реальном времени...")
            csi_mode(interface, target_mac, channel)
            input("\nНажмите Enter для продолжения...")
        
        elif choice == "4":
            return
        
        else:
            print("Неверный выбор")

def auto_room_tracking(interface, router_mac, channel=6):
    """Автоматическое создание модели комнаты и отслеживание движения"""
    # Настройка адаптера
    if not setup_interface(interface, "monitor"):
        return
        
    target_mac_bin = binascii.unhexlify(router_mac.replace(':', ''))
    sock = setup_csi_socket(interface, target_mac_bin, channel)
    if sock is None:
        return
    
    # Инициализация модели комнаты
    room_radius = 5.0  # Начальный предполагаемый радиус комнаты (метры)
    router_position = np.array([0.0, 0.0])  # Роутер в центре
    object_position = np.array([0.0, 0.0])  # Начальная позиция объекта
    walls = []  # Стены комнаты
    
    # Инициализация визуализации
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Автоматическая модель комнаты")
    ax.set_xlabel("X (м)")
    ax.set_ylabel("Y (м)")
    ax.grid(True)
    
    # Отрисовка роутера
    router_dot = ax.scatter([router_position[0]], [router_position[1]], 
                          s=200, c='red', marker='s', label='Роутер')
    
    # Отрисовка объекта
    object_dot, = ax.plot([object_position[0]], [object_position[1]], 
                        'go', markersize=12, label='Объект')
    
    # Отрисовка стен
    wall_lines = []
    
    # История позиций
    trajectory_x = []
    trajectory_y = []
    trajectory, = ax.plot(trajectory_x, trajectory_y, 'b-', alpha=0.5)
    
    # Начальные значения
    last_amps = None
    last_time = time.time()
    max_distance = 0.0  # Максимальное зафиксированное расстояние
    
    # Физические параметры
    motion_decay = 0.92  # Затухание движения
    max_speed = 1.5      # Максимальная скорость (м/сек)
    
    print("\nАвтоматическое создание модели комнаты...")
    print("Роутер находится в центре (0, 0)")
    print("Двигайтесь по комнате для построения модели")
    
    try:
        while not STOP_EVENT.is_set():
            data = sock.recv(2048)
            amps, header = parse_csi_packet(data)
            if amps is None:
                continue
            
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            if last_amps is None:
                last_amps = amps
                continue
            
            # Анализ изменений CSI
            diff = amps - last_amps
            last_amps = amps
            
            # Определение направления движения
            left_diff = np.mean(diff[:32])
            right_diff = np.mean(diff[32:])
            front_diff = np.mean(diff[16:48])
            
            # Преобразование в вектор движения
            movement_vector = np.array([
                (right_diff - left_diff) * 0.2,  # X-компонента
                front_diff * 0.3                 # Y-компонента (к роутеру)
            ])
            
            # Фильтрация шумов
            movement_norm = np.linalg.norm(movement_vector)
            if movement_norm < 0.5:
                movement_vector *= 0
            else:
                movement_vector = movement_vector / movement_norm * min(movement_norm, 3.0)
            
            # Обновление позиции объекта
            velocity = movement_vector * dt
            speed = np.linalg.norm(velocity)
            if speed > max_speed:
                velocity = velocity / speed * max_speed
            
            object_position += velocity
            
            # Ограничение позиции в пределах комнаты
            distance = np.linalg.norm(object_position)
            if distance > room_radius * 0.9:
                # "Отскок" от стены
                object_position = object_position / distance * room_radius * 0.9
            
            # Обновление максимального расстояния
            if distance > max_distance:
                max_distance = distance
                room_radius = max_distance * 1.2  # +20% запаса
                
                # Обновление стен
                walls = []
                for angle in np.linspace(0, 2*np.pi, 16):  # 16 точек по кругу
                    x = room_radius * np.cos(angle)
                    y = room_radius * np.sin(angle)
                    walls.append((x, y))
            
            # Обновление траектории
            trajectory_x.append(object_position[0])
            trajectory_y.append(object_position[1])
            
            # Ограничение истории
            if len(trajectory_x) > 100:
                trajectory_x.pop(0)
                trajectory_y.pop(0)
            
            # Обновление визуализации ---
            
            # Очистка предыдущих стен
            for line in wall_lines:
                line.remove()
            wall_lines = []
            
            # Отрисовка новых стен
            for i in range(len(walls)):
                x1, y1 = walls[i]
                x2, y2 = walls[(i+1) % len(walls)]
                line, = ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
                wall_lines.append(line)
            
            # Обновление позиции объекта
            object_dot.set_data([object_position[0]], [object_position[1]])
            
            # Обновление траектории
            trajectory.set_data(trajectory_x, trajectory_y)
            
            # Обновление текста
            if 'dist_text' in locals():
                dist_text.remove()
            dist_text = ax.text(0.02, 0.95, 
                               f"Расстояние: {distance:.1f} м\nРадиус комнаты: {room_radius:.1f} м",
                               transform=ax.transAxes, fontsize=12,
                               bbox=dict(facecolor='white', alpha=0.7))
            
            # Автомасштабирование
            margin = room_radius * 0.2
            ax.set_xlim(-room_radius - margin, room_radius + margin)
            ax.set_ylim(-room_radius - margin, room_radius + margin)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            
    except KeyboardInterrupt:
        print("\nОтслеживание остановлено")
    except Exception as e:
        print(f"\nОшибка в автоматическом моделировании комнаты: {str(e)}")
        traceback.print_exc()
    finally:
        sock.close()
        setup_interface(interface, "managed")
        plt.ioff()
        if not STOP_EVENT.is_set():
            plt.show()

def main_menu():
    """Главное меню программы"""
    global STOP_EVENT, CURRENT_MODE
    
    interface = None
    target_mac = None
    channel = 6
    
    while True:
        STOP_EVENT.clear()
        print_header("ГЛАВНОЕ МЕНЮ WI-FI SENSING")
        
        status = f"{TerminalColors.WARNING}Не выбрано{TerminalColors.ENDC}"
        if interface:
            iface_status = f"{TerminalColors.OKGREEN}{interface}{TerminalColors.ENDC}"
            if target_mac:
                target_status = f"{TerminalColors.OKGREEN}{target_mac} (канал {channel}){TerminalColors.ENDC}"
            else:
                target_status = f"{TerminalColors.WARNING}Не выбрана{TerminalColors.ENDC}"
        else:
            iface_status = status
            target_status = status
        
        print(f"1. Выбрать адаптер: {iface_status}")
        print(f"2. Выбрать сеть: {target_status}")
        print("3. Режим CSI (расширенный мониторинг)")
        print("4. Режим RSSI (базовый мониторинг)")
        print("5. Система Who-Fi (идентификация)")
        print("6. Автоматическая модель комнаты")
        print("0. Выход")
        
        choice = input("\nВыберите действие: ")
        
        if choice == "1":
            interface = select_interface()
            target_mac = None  # Сброс выбранной сети
            
        elif choice == "2":
            if not interface:
                print("Сначала выберите адаптер!")
                time.sleep(1)
                continue
            target_mac, channel = select_network(interface)
            
        elif choice == "3":
            if not interface or not target_mac:
                print("Сначала выберите адаптер и сеть!")
                time.sleep(1)
                continue
                
            if not ADAPTER_COMPATIBILITY.get(interface, {}).get('csi', False):
                print(f"{TerminalColors.WARNING}Внимание: Ваш адаптер может не поддерживать CSI режим!{TerminalColors.ENDC}")
                print("Попробуйте использовать адаптеры Intel 5300, Atheros AR9xxx или аналоги")
                confirm = input("Продолжить? (y/N): ")
                if confirm.lower() != 'y':
                    continue
            
            # Запуск CSI режима в отдельном потоке
            STOP_EVENT.clear()
            csi_thread = threading.Thread(target=csi_mode, args=(interface, target_mac, channel))
            csi_thread.daemon = True
            csi_thread.start()
            
            # Ожидание завершения или команды остановки
            while csi_thread.is_alive():
                if input("\nВведите 'q' для остановки мониторинга: ") == 'q':
                    STOP_EVENT.set()
                    csi_thread.join(timeout=2)
                    break
                time.sleep(0.1)
            
        elif choice == "4":
            if not interface or not target_mac:
                print("Сначала выберите адаптер и сеть!")
                time.sleep(1)
                continue
                
            # Запуск RSSI режима в отдельном потоке
            STOP_EVENT.clear()
            rssi_thread = threading.Thread(target=rssi_mode, args=(interface, target_mac))
            rssi_thread.daemon = True
            rssi_thread.start()
            
            # Ожидание завершения или команды остановки
            while rssi_thread.is_alive():
                if input("\nВведите 'q' для остановки мониторинга: ") == 'q':
                    STOP_EVENT.set()
                    rssi_thread.join(timeout=2)
                    break
                time.sleep(0.1)
            
        elif choice == "5":
            if not interface or not target_mac:
                print("Сначала выберите адаптер и сеть!")
                time.sleep(1)
                continue
            who_fi_menu(interface, target_mac, channel)
            
        elif choice == "6":
            if not interface or not target_mac:
                print("Сначала выберите адаптер и сеть!")
                time.sleep(1)
                continue
                
            # Запуск автотрекинга
            STOP_EVENT.clear()
            auto_thread = threading.Thread(target=auto_room_tracking, args=(interface, target_mac, channel))
            auto_thread.daemon = True
            auto_thread.start()
            
            # Ожидание завершения или команды остановки
            while auto_thread.is_alive():
                if input("\nВведите 'q' для остановки: ") == 'q':
                    STOP_EVENT.set()
                    auto_thread.join(timeout=2)
                    break
                time.sleep(0.1)
            
        elif choice == "0":
            print("Выход...")
            exit(0)
            
        else:
            print("Неверный выбор")

def main():
    global WHOFI_TRAINING_MODE, WHOFI_CURRENT_USER, WHOFI_ALERT_UNKNOWN
    
    # Проверка прав администратора
    if os.geteuid() != 0:
        print(f"{TerminalColors.FAIL}Требуются права root! Запустите скрипт с помощью sudo.{TerminalColors.ENDC}")
        exit(1)
    
    # Проверка необходимых инструментов
    required_tools = ['iw', 'iwconfig'] if platform.system() == "Linux" else []
    for tool in required_tools:
        if shutil.which(tool) is None:
            print(f"{TerminalColors.FAIL}Не найдена необходимая утилита: {tool}{TerminalColors.ENDC}")
            exit(1)
    
    # Обработка аргументов командной строки
    parser = argparse.ArgumentParser(description='Wi-Fi мониторинг с Who-Fi идентификацией')
    parser.add_argument('-i', '--interface', help='Сетевой интерфейс')
    parser.add_argument('-m', '--mac', help='MAC-адрес роутера')
    parser.add_argument('-c', '--channel', type=int, help='Номер канала')
    parser.add_argument('--who-fi-train', metavar='NAME', help='Обучить нового пользователя Who-Fi')
    parser.add_argument('--no-alert', action='store_true', help='Отключить оповещения о незнакомцах')
    args = parser.parse_args()
    
    if args.who_fi_train:
        WHOFI_TRAINING_MODE = True
        WHOFI_CURRENT_USER = args.who_fi_train
        if not args.interface or not args.mac:
            print("Для обучения Who-Fi необходимо указать интерфейс и MAC-адрес")
            exit(1)
        who_fi_train_new_user(args.interface, args.mac, args.who_fi_train, args.channel or 6)
        exit(0)
    
    if args.no_alert:
        WHOFI_ALERT_UNKNOWN = False
    
    # Запуск интерактивного меню
    main_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрограмма завершена")
        exit(0)
    except Exception as e:
        print(f"\n{TerminalColors.FAIL}Критическая ошибка: {str(e)}{TerminalColors.ENDC}")
        traceback.print_exc()
        exit(1)