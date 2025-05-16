#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
#import ccml_optimized as ccml
import ccml
import pyaudio
import time
import os
import datetime
from matplotlib import style
style.use('ggplot')

CHUNK = 4096
FORMAT = pyaudio.paUInt8
CHANNELS = 1
RATE = 44100
DEVICE_INDEX = None

N_BITS_PER_CHUNK = 13000000  # wymagane dla testów NIST
INTERVAL_SECONDS = 120
SAMPLE_TIME = 5
OUTPUT_DIR = "output"

MAX_BUFFER_SIZE = 20000000
BUFFER_SAFETY_MARGIN = 0.1

# Utwórz katalog wyjściowy jeśli nie istnieje
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

global_lsb3_buffer = np.array([], dtype=np.uint8)
global_raw_buffer = np.array([], dtype=np.uint8)
def get_timestamp_filename(prefix, extension):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{OUTPUT_DIR}/{prefix}_{timestamp}.{extension}"

def record_audio_samples(p, device_index, duration_seconds):
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   input_device_index=device_index,
                   frames_per_buffer=CHUNK)
    
    n_chunks = int(RATE * duration_seconds / CHUNK)
    frames = []
    
    for i in range(n_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.uint8))
    stream.stop_stream()
    stream.close()
    
    raw_samples = np.concatenate(frames)
    print(f"Zebrano {len(raw_samples)} próbek audio")
    return raw_samples

def process_audio_samples(timestamp):
    global global_lsb3_buffer
    global global_raw_buffer
    
    timestamp_short = datetime.datetime.now().strftime("%H%M%S")
    required_samples = calculate_required_samples()
    
    if len(global_lsb3_buffer) < required_samples:
        print(f"OSTRZEŻENIE: Niewystarczająca liczba próbek ({len(global_lsb3_buffer):,}/{required_samples:,})")
        print(f"Spróbuj ponownie później, gdy zbierzemy więcej próbek")
        return None, None
    
    try:
        lsb3_to_use = global_lsb3_buffer[:required_samples].copy()
        
        raw_samples_to_use = min(required_samples, len(global_raw_buffer))
        raw_to_use = global_raw_buffer[:raw_samples_to_use].copy() if raw_samples_to_use > 0 else np.array([], dtype=np.uint8)
        
        lsb3_to_use = lsb3_to_use.astype(np.uint8)
        
        bit_stream = np.vstack([((lsb3_to_use >> i) & 1) for i in [0,1,2]]).T.flatten()[:N_BITS_PER_CHUNK]
        
        bit_bytes = np.packbits(bit_stream)
        source_bin_filename = f"{OUTPUT_DIR}/source_{timestamp_short}.bin"
        
        with open(source_bin_filename, 'wb') as f:
            f.write(bit_bytes)
        
        print(f"Wygenerowano {len(bit_stream):,} bitów, zapisano do {source_bin_filename}")
        
        if len(global_lsb3_buffer) >= required_samples:
            global_lsb3_buffer = global_lsb3_buffer[required_samples:]
        else:
            global_lsb3_buffer = np.array([], dtype=np.uint8)
        
        if len(global_raw_buffer) >= raw_samples_to_use:
            global_raw_buffer = global_raw_buffer[raw_samples_to_use:]
        else:
            global_raw_buffer = np.array([], dtype=np.uint8)
            
    except Exception as e:
        print(f"BŁĄD podczas przetwarzania próbek: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # Generowanie wykresów
    generate_plots(raw_to_use, lsb3_to_use, timestamp_short)
    
    # Obliczanie entropii
    calculate_entropy(lsb3_to_use)
    
    # Przetwarzanie CCML
    post_bin_filename = f"{OUTPUT_DIR}/post_{timestamp_short}.bin"
    plot_filename = f"{OUTPUT_DIR}/ccml_dist_{timestamp_short}.png"
    ccml.run_ccml(
        filename=source_bin_filename,
        output_filename=post_bin_filename,
        N_target_bits=N_BITS_PER_CHUNK,
        plot_filename=plot_filename
    )
    

    
    return source_bin_filename, post_bin_filename

def calculate_required_samples():
    return int(np.ceil(N_BITS_PER_CHUNK / 3))

# Parametry kontroli bufora
MAX_BUFFER_MULTIPLIER = 2.0
BUFFER_TRIM_THRESHOLD = 1.8

def add_samples_to_buffer(raw_samples):
    global global_lsb3_buffer
    global global_raw_buffer
    
    lsb3 = raw_samples & 0b00000111
    required_samples = calculate_required_samples()
    
    global_lsb3_buffer = np.concatenate([global_lsb3_buffer, lsb3])
    global_raw_buffer = np.concatenate([global_raw_buffer, raw_samples])
    
    max_buffer_size = int(required_samples * MAX_BUFFER_MULTIPLIER)
    if len(global_lsb3_buffer) > max_buffer_size:
        excess = len(global_lsb3_buffer) - max_buffer_size
        global_lsb3_buffer = global_lsb3_buffer[excess:]
        print(f"Bufor LSB przycięty o {excess:,} próbek (przekroczenie limitu {MAX_BUFFER_MULTIPLIER}x)")
    
    if len(global_raw_buffer) > len(global_lsb3_buffer):
        global_raw_buffer = global_raw_buffer[-len(global_lsb3_buffer):]
    
    buffer_percentage = min(100, len(global_lsb3_buffer) / required_samples * 100)
    print(f"Bufor zawiera {len(global_lsb3_buffer):,} próbek 3LSB ({len(global_lsb3_buffer)*3:,} bitów)")
    print(f"Potrzeba {required_samples:,} próbek ({N_BITS_PER_CHUNK:,} bitów)")
    print(f"Postęp: {buffer_percentage:.1f}% (limit bufora: {MAX_BUFFER_MULTIPLIER:.1f}x)")

# Funkcja do generowania wykresów
def generate_plots(raw, lsb3, timestamp):
    plt.figure(figsize=(10, 6))
    plt.hist(raw, bins=256, color='blue', alpha=0.7)
    plt.title("Histogram próbek audio")
    plt.xlabel("Wartość próbki")
    plt.ylabel("Częstotliwość")
    plt.yscale('log')
    plt.savefig(f"{OUTPUT_DIR}/raw_hist_{timestamp}.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    counts_raw = Counter(raw)
    probabilities_raw = np.array([counts_raw[i]/len(raw) if len(raw) > 0 else 0 for i in range(256)])
    plt.bar(range(256), probabilities_raw, color='darkslateblue', width=1.0)
    plt.title("Empiryczny rozkład prawdopodobieństwa próbek audio")
    plt.xlabel("Wartość próbki (x)")
    plt.ylabel("Częstotliwość występowania (p_i)")
    plt.xlim([-0.5, 255.5])
    plt.savefig(f"{OUTPUT_DIR}/raw_dist_{timestamp}.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.hist(lsb3, bins=8, range=(-0.5, 7.5), color='purple', alpha=0.7, rwidth=0.8)
    plt.title("Histogram wartości 3 LSB")
    plt.xlabel("Wartość 3 LSB (0-7)")
    plt.ylabel("Częstotliwość")
    plt.xticks(range(8))
    plt.yscale('log')
    plt.savefig(f"{OUTPUT_DIR}/lsb3_hist_{timestamp}.png")
    plt.close()

# Funkcja do obliczania entropii
def calculate_entropy(lsb3):
    cnt = Counter(lsb3)
    probabilities = np.array([cnt[i]/len(lsb3) for i in range(8)])
    
    entropy_terms = []
    for p_i in probabilities:
        if p_i > 0:
            entropy_terms.append(p_i * np.log2(p_i))
    
    if not entropy_terms:
        H = 0.0
    else:
        H = -np.sum(entropy_terms)
    
    print(f"Entropia 3 LSB: {H:.4f} bitów na symbol")
    return H

# Główna funkcja nagrywania i przetwarzania strumienia audio
def run_continuous_trng():
    p = pyaudio.PyAudio()
    
    print("\nDostępne urządzenia wejściowe audio:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            print(f"Index {i}: {device_info['name']}")
    
    device_index = DEVICE_INDEX
    if device_index is None:
        try:
            device_index_str = input("\nWybierz indeks urządzenia wejściowego (Enter dla domyślnego): ")
            if device_index_str.strip():
                device_index = int(device_index_str)
                device_info = p.get_device_info_by_index(device_index)
                print(f"Wybrano: {device_info['name']}")
        except (ValueError, IOError):
            print("Nieprawidłowy indeks, używam urządzenia domyślnego.")
    
    print("\nNaciskaj Ctrl+C aby zakończyć nagrywanie.")
    print(f"Generowanie plików wyjściowych co {INTERVAL_SECONDS} sekund...")
    print(f"Ciągłe nagrywanie po {SAMPLE_TIME} sekund na cykl...")
    
    last_processing_time = 0
    required_samples = calculate_required_samples()
    
    try:
        while True:
            cycle_start_time = time.time()
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            raw_samples = record_audio_samples(p, device_index, SAMPLE_TIME)
            add_samples_to_buffer(raw_samples)
            
            time_since_last_processing = time.time() - last_processing_time
            buffer_ready = len(global_lsb3_buffer) >= required_samples
            
            if time_since_last_processing >= INTERVAL_SECONDS:
                if buffer_ready:
                    print(f"\n--- Przetwarzanie danych [{current_time}] ---")
                    source_bin, post_bin = process_audio_samples(current_time)
                    
                    if source_bin is not None:
                        last_processing_time = time.time()
                else:
                    processing_pct = len(global_lsb3_buffer) / required_samples * 100
                    print(f"\n--- Czekam na więcej danych ({processing_pct:.1f}%) ---")
            
            cycle_duration = time.time() - cycle_start_time
            
            target_cycle_time = max(0.5, SAMPLE_TIME * 0.2)
            short_sleep = max(0.1, min(target_cycle_time - cycle_duration, SAMPLE_TIME * 0.5))
            
            print(f"Cykl nagrywania: {cycle_duration:.2f}s, następny cykl za {short_sleep:.2f}s")
            time.sleep(short_sleep)
    
    except KeyboardInterrupt:
        print("\nPrzerwano przez użytkownika. Zamykanie...")
    finally:
        p.terminate()
        print("Nagrywanie zakończone.")

if __name__ == "__main__":
    print("=== Rozpoczynam ciągły generator liczb losowych ===")
    print(f"Częstotliwość próbkowania: {RATE} Hz")
    print(f"Format: {FORMAT}, Kanały: {CHANNELS}")
    print(f"Wymagana liczba bitów na plik: {N_BITS_PER_CHUNK:,}")
    print(f"Interwał generowania plików: {INTERVAL_SECONDS} sekund")
    print(f"Czas cyklu nagrywania: {SAMPLE_TIME} sekund")
    print(f"Pliki wyjściowe będą zapisywane w katalogu: '{OUTPUT_DIR}'")
    
    run_continuous_trng()
