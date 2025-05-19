#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import ccml
import pyaudio
import time
import os
import sys
import datetime
import subprocess
from matplotlib import style
style.use('ggplot')

CHUNK = 8192
FORMAT = pyaudio.paUInt8
CHANNELS = 1
RATE = 44100
DEVICE_INDEX = None

N_BITS_PER_CHUNK = 104_000_000  # 13MB (104 miliony bitów)
INTERVAL_SECONDS = 300  # 5 minut między generowaniem plików
SAMPLE_TIME = 15  # czas nagrywania pojedynczego fragmentu
OUTPUT_DIR = "output"

MAX_BUFFER_SIZE = 160000000
BUFFER_SAFETY_MARGIN = 0.1


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

global_lsb3_buffer = np.array([], dtype=np.uint8)
global_raw_buffer = np.array([], dtype=np.uint8)
last_entropy_source = 0.0
last_entropy_post = 0.0
last_output_file = ""

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
        
        # Przekształcenie 3 LSB w strumień bitów
        bit_stream = np.vstack([((lsb3_to_use >> i) & 1) for i in [0,1,2]]).T.flatten()[:N_BITS_PER_CHUNK]
        
        # Pakowanie bitów w bajty do zapisania w pliku
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
    
    # Generowanie wykresów dla surowych danych audio
    generate_raw_plots(raw_to_use, timestamp_short)
    
    # Obliczanie entropii dla pliku source.bin
    source_entropy = calculate_source_entropy(bit_stream)
    
    # Przetwarzanie CCML
    post_bin_filename = f"{OUTPUT_DIR}/post_{timestamp_short}.bin"
    plot_filename = f"{OUTPUT_DIR}/ccml_dist_{timestamp_short}.png"
    generated_bits = ccml.run_ccml(
        filename=source_bin_filename,
        output_filename=post_bin_filename,
        N_target_bits=N_BITS_PER_CHUNK,
        plot_filename=plot_filename,
        verbose=False  # Mniej komunikatów diagnostycznych
    )
    
    # Zapisz informacje o ostatnim pliku i entropii
    global last_output_file, last_entropy_post
    last_output_file = post_bin_filename
    if hasattr(ccml, 'last_entropy'):
        last_entropy_post = ccml.last_entropy
    
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
    
    if len(global_raw_buffer) > len(global_lsb3_buffer):
        global_raw_buffer = global_raw_buffer[-len(global_lsb3_buffer):]

# Funkcja do generowania wykresów dla surowych danych audio
def generate_raw_plots(raw, timestamp):
    # Histogram surowych próbek audio
    plt.figure(figsize=(10, 6))
    
    # Sprawdź, czy są jakiekolwiek dane
    if len(raw) == 0:
        print("Ostrzeżenie: Brak danych próbek audio do wygenerowania histogramu.")
        plt.text(128, 1, "Brak danych", horizontalalignment='center', fontsize=14)
        plt.ylim([0.1, 10])
    else:
        plt.hist(raw, bins=256, color='blue', alpha=0.7)
        plt.yscale('log')
    
    plt.title("Histogram próbek audio")
    plt.xlabel("Wartość próbki")
    plt.ylabel("Częstotliwość")
    plt.xlim([-0.5, 255.5])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/raw_hist_{timestamp}.png", dpi=150)
    plt.close()

    # Rozkład prawdopodobieństwa surowych próbek audio
    plt.figure(figsize=(10, 6))
    
    # Sprawdź, czy są jakiekolwiek dane
    if len(raw) == 0:
        print("Ostrzeżenie: Brak danych próbek audio do wygenerowania rozkładu.")
        plt.text(128, 0.5, "Brak danych", horizontalalignment='center', fontsize=14)
        plt.ylim([0, 1.0])
    else:
        counts_raw = Counter(raw)
        probabilities_raw = np.array([counts_raw.get(i, 0)/len(raw) for i in range(256)])
        plt.bar(range(256), probabilities_raw, color='darkslateblue', width=1.0)
        
        # Ustaw odpowiedni zakres osi Y, aby dane były widoczne
        max_prob = probabilities_raw.max()
        if max_prob > 0:
            plt.ylim([0, max_prob * 1.1])  # Dodaj 10% marginesu na górze
        else:
            plt.ylim([0, 0.1])  # Domyślna wartość jeśli nie ma danych
    
    plt.title("Empiryczny rozkład prawdopodobieństwa próbek audio")
    plt.xlabel("Wartość próbki (x)")
    plt.ylabel("Częstotliwość występowania (p_i)")
    plt.xlim([-0.5, 255.5])
    
    # Dodaj siatkę dla lepszej czytelności
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/raw_dist_{timestamp}.png", dpi=150)
    plt.close()

# Funkcja do obliczania entropii dla source.bin (strumień bitów)
def calculate_source_entropy(bit_stream):
    global last_entropy_source
    
    # Konwersja strumienia bitów na bajty dla analizy
    bit_count = len(bit_stream)
    padding = (8 - bit_count % 8) % 8
    padded_bits = np.pad(bit_stream, (0, padding), 'constant')
    
    # Przekształć strumień bitów w bajty dla analizy entropii
    byte_array = np.packbits(padded_bits.reshape(-1, 8))
    
    # Analizuj rozkład bajtów i generuj wykresy
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    generate_source_plots(byte_array, timestamp)
    
    # Obliczanie entropii na poziomie bajtów
    cnt = Counter(byte_array)
    probabilities = np.array([cnt.get(i, 0)/len(byte_array) for i in range(256)])
    
    # Obliczanie entropii z logarytmem o podstawie 2 (informacja w bitach)
    entropy_terms = []
    for p_i in probabilities:
        if p_i > 0:
            entropy_terms.append(p_i * np.log2(p_i))
    
    if not entropy_terms:
        H = 0.0
    else:
        H = -np.sum(entropy_terms)
    
    # Normalizacja do bitów na symbol (bajt)
    H_normalized = H / 8.0
    
    # Obliczanie entropii bezpośrednio na poziomie bitów
    bit_array = np.unpackbits(byte_array)
    bit_counts = Counter(bit_array)
    p_0 = bit_counts.get(0, 0)/len(bit_array)
    p_1 = bit_counts.get(1, 0)/len(bit_array)
    
    H_bits = 0.0
    if p_0 > 0:
        H_bits -= p_0 * np.log2(p_0)
    if p_1 > 0:
        H_bits -= p_1 * np.log2(p_1)
    
    global last_entropy_source
    last_entropy_source = H_normalized  # Zachowujemy znormalizowaną entropię bajtową
    
    # Wyświetl obie wartości entropii dla porównania
    print(f"Entropia source.bin: {H:.4f} bitów/symbol (bajt)")
    print(f"Entropia source.bin: {H_normalized:.4f} bitów/bit")
    print(f"Entropia bitowa: {H_bits:.4f} bitów/bit (idealna: 1.0)")
    
    # Zwróć pełną wartość entropii (bitów/bajt) a nie znormalizowaną
    return H

def generate_source_plots(byte_array, timestamp):
    # Histogram rozkładu prawdopodobieństwa bajtów
    plt.figure(figsize=(10,6))
    plt.hist(byte_array,
             bins=256,
             range=(0,255),
             density=True,
             color='darkslateblue',
             alpha=0.8)
    plt.title("Empiryczny rozkład bajtów w source.bin")
    plt.xlabel("Wartość bajtu")
    plt.ylabel("Prawdopodobieństwo p_i")
    plt.xlim(-0.5, 255.5)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/source_dist_{timestamp}.png", dpi=150)
    plt.close()

    # Rozkład bitów 0/1
    bit_array = np.unpackbits(byte_array, bitorder='big')
    p0 = np.mean(bit_array==0)
    p1 = 1 - p0
    plt.figure(figsize=(6,4))
    plt.bar([0,1],[p0,p1], color=['teal','forestgreen'])
    plt.title("Rozkład bitów w source.bin")
    plt.xticks([0,1])
    plt.ylabel("Prawdopodobieństwo")
    plt.ylim(0,1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/source_bits_{timestamp}.png", dpi=150)
    plt.close()

    counts = Counter(byte_array)
    probs = np.array([counts[i]/len(byte_array) for i in range(256)])
    return probs


last_entropy_source = 0.0
last_entropy_post = 0.0
last_output_file = ""



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
            
            print(f"\nRozpoczęcie nagrywania próbek audio na {SAMPLE_TIME} sekund...")
            raw_samples = record_audio_samples(p, device_index, SAMPLE_TIME)
            
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Nagłówek informacyjny
            print(f"=== TRNG - Status [{datetime.datetime.now().strftime('%H:%M:%S')}] ===")
            print(f"Urządzenie audio: {p.get_device_info_by_index(device_index)['name'] if device_index is not None else 'domyślne'}")
            
            # Dodaj próbki do bufora
            add_samples_to_buffer(raw_samples)
            
            time_since_last_processing = time.time() - last_processing_time
            buffer_ready = len(global_lsb3_buffer) >= required_samples
            
            print("\n--- Informacje o entropii ---")
            bit_per_symbol_source = last_entropy_source * 8  # Konwertuj z bit/bit na bit/symbol
            print(f"Entropia source.bin: {bit_per_symbol_source:.4f} bitów/symbol | {last_entropy_source:.4f} bitów/bit")
            
            # Informacja o ostatnim pliku
            if last_output_file:
                print(f"Ostatni plik wyjściowy: {os.path.basename(last_output_file)}")
                if last_entropy_post > 0:
                    post_bit_per_bit = last_entropy_post / 8  # Konwertuj z bit/symbol na bit/bit
                    print(f"Entropia po CCML: {last_entropy_post:.4f} bitów/symbol | {post_bit_per_bit:.4f} bitów/bit")
            
            buffer_percentage = min(100, len(global_lsb3_buffer) / required_samples * 100)
            print(f"\n--- Status bufora ---")
            print(f"Bufor zawiera {len(global_lsb3_buffer):,} próbek 3LSB ({len(global_lsb3_buffer)*3:,} bitów)")
            print(f"Potrzeba {required_samples:,} próbek ({N_BITS_PER_CHUNK:,} bitów)")
            print(f"Postęp: {buffer_percentage:.1f}% (limit bufora: {MAX_BUFFER_MULTIPLIER:.1f}x)")
            print(f"Czas od ostatniego wygenerowania pliku: {int(time_since_last_processing)} s / {INTERVAL_SECONDS} s")
            
            max_buffer_size = int(required_samples * MAX_BUFFER_MULTIPLIER)
            if len(global_lsb3_buffer) >= max_buffer_size:
                print(f"Status: Bufor osiągnął maksymalną pojemność ({max_buffer_size:,} próbek)")
            
            # Przetwarzanie
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
            
            target_cycle_time = max(1.0, SAMPLE_TIME * 0.5)
            short_sleep = max(1.0, min(target_cycle_time - cycle_duration, SAMPLE_TIME * 0.5))
            
            print(f"\nCykl nagrywania: {cycle_duration:.2f}s, następny cykl za {short_sleep:.2f}s")
            print("\nNaciśnij Ctrl+C aby zakończyć")
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
