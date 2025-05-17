#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import sys
import os
from numba import njit, prange

ALPHA = 1.99999   
last_entropy = 0.0

@njit
def piecewise_map(x: float, alpha: float = ALPHA) -> float:
    """Mapowanie kawałkowe (tent map) z parametrem α"""
    return alpha * x if x <= 0.5 else alpha * (1.0 - x)

@njit
def ccml_step(states: np.ndarray, alpha: float = ALPHA, epsilon: float = 0.05) -> np.ndarray:
    """Jeden krok algorytmu CCML"""
    L = states.size
    new_states = np.empty_like(states)
    for i in range(L):
        left   = piecewise_map(states[(i - 1) % L], alpha)
        center = piecewise_map(states[i], alpha)
        right  = piecewise_map(states[(i + 1) % L], alpha)
        new_states[i] = ((1.0 - epsilon) * center + (epsilon * 0.5) * (left + right))
    return new_states

def print_ascii_histogram(hist_dict, max_width=50):
    max_value = max(hist_dict.values())
    for value, count in sorted(hist_dict.items()):
        bar_length = int(count / max_value * max_width)
        print(f"{value:4}: {'#' * bar_length} ({count})")

def get_binary_string(file_path, bit_count=1000):
    """Odczytaj pierwsze bit_count bitów z pliku"""
    with open(file_path, 'rb') as f:
        data = f.read((bit_count + 7) // 8)
        
    binary_string = ""
    for byte in data:
        binary_string += format(byte, '08b')
    
    return binary_string[:bit_count]

def split_sequence(seq, n=2):
    """Podziel sekwencję na n-gramy"""
    return ["".join(seq[i:i+n]) for i in range(0, len(seq), n)]

@njit
def calculate_new_bits(n_chaotic_bits, n_target_bits):
    """Oblicz ile bitów "zapasowych" musimy wygenerować"""
    return n_target_bits + int(n_target_bits * 0.1)  # 10% zapas

@njit
def get_bit_from_chaotic_state(x):
    return 1 if x > 0.5 else 0

@njit
def vectorized_get_bit(states):
    return np.where(states > 0.5, 1, 0)

@njit(parallel=True)
def generate_bits_from_ccml(states, n_steps, n_bits, alpha=ALPHA, epsilon=0.05):
    """Generowanie bitów z układu CCML"""
    bits = np.zeros(n_bits, dtype=np.int8)
    current_states = states.copy()
    bit_index = 0
    
    for _ in range(n_steps):
        if bit_index >= n_bits:
            break
            
        current_states = ccml_step(current_states, alpha, epsilon)
        
        state_bits = vectorized_get_bit(current_states)
        
        bits_to_copy = min(len(state_bits), n_bits - bit_index)
        bits[bit_index:bit_index + bits_to_copy] = state_bits[:bits_to_copy]
        bit_index += bits_to_copy
    
    return bits

def run_ccml(filename, output_filename, N_target_bits=1000000, plot_filename=None, verbose=True):
    """Główna funkcja do uruchamiania algorytmu CCML na pliku wejściowym"""
    if not os.path.exists(filename):
        print(f"BŁĄD: Plik {filename} nie istnieje.")
        return
        
    with open(filename, "rb") as f:
        data_bytes = f.read()
    
    bits = []
    for byte in data_bytes:
        for bit_pos in range(8):
            bit = (byte >> bit_pos) & 1
            bits.append(bit)
    
    if verbose:
        print(f"Wczytano {len(bits)} bitów z pliku {filename}")
    
    if len(bits) < 1000:
        print(f"BŁĄD: Za mało bitów ({len(bits)} < 1000).")
        return
    
    # Przygotowanie początkowego stanu sieci CCML
    L = 128  # Rozmiar sieci (liczba stanów)
    n_chaotic_bits = calculate_new_bits(L, N_target_bits)  # Ile bitów wygenerować
    
    # Inicjalizacja stanów CCML za pomocą bitów ze źródła entropii
    initial_states = np.zeros(L)
    for i in range(L):
        s = 0.0
        for j in range(10):  # Używamy 10 bitów dla każdego stanu
            idx = (i * 10 + j) % len(bits)
            s = s/2 + 0.5*bits[idx]
        initial_states[i] = s
    
    if verbose:
        print(f"Zainicjalizowano {L} stanów CCML używając {L*10} bitów wejściowych")
    
    # Parametry CCML
    ALPHA = 1.99999   # Parametr chaotyczny (bardzo bliski 2)
    EPSILON = 0.05    # Parametr sprzężenia (coupling)
    
    # Odpuszczenie stanów początkowych (transient)
    current_states = initial_states.copy()
    for _ in range(100):
        current_states = ccml_step(current_states, ALPHA, EPSILON)
    
    # Generowanie bitów
    n_steps = (n_chaotic_bits + L - 1) // L
    
    if verbose:
        print(f"Generowanie {n_chaotic_bits} bitów w {n_steps} krokach CCML...")
    
    generated_bits = generate_bits_from_ccml(
        current_states, n_steps, n_chaotic_bits, ALPHA, EPSILON)
    
    if verbose:
        print(f"Wygenerowano {len(generated_bits)} bitów")
    
    # Upewnij się, że mamy dokładnie N_target_bits
    final_bits = generated_bits[:N_target_bits]
    
    # Konwertuj bity na bajty i zapisz do pliku wyjściowego
    output_bytes = np.packbits(final_bits)
    with open(output_filename, "wb") as f:
        f.write(output_bytes.tobytes())
    
    if verbose:
        print(f"Zapisano {len(final_bits)} bitów ({len(output_bytes)} bajtów) do pliku {output_filename}")
    
    # Analiza i wizualizacja rozkładu
    if plot_filename:
        analyze_distribution(final_bits, plot_filename)
    
    return final_bits

def analyze_distribution(bits, plot_filename=None):
    """Analizuj rozkład bajtów w wyjściowych danych"""
    if bits.size == 0:
        print("Brak bitów wyjściowych do wygenerowania wykresu.")
        return None
    
    # Przygotuj bity do konwersji na bajty
    num_total_bits = bits.size
    padded_len_for_stats = ((num_total_bits + 7) // 8) * 8
    padded_bits_for_stats = np.pad(bits, (0, padded_len_for_stats - num_total_bits), constant_values=0)
    byte_array_for_stats = np.packbits(padded_bits_for_stats)
    
    if byte_array_for_stats.size == 0:
        print("Brak danych bajtowych do wygenerowania wykresu.")
        return None
    
    # Oblicz częstości i prawdopodobieństwa
    counts = Counter(byte_array_for_stats)
    probs = np.array([counts.get(i, 0) / byte_array_for_stats.size for i in range(256)])
    
    # Rysuj histogram
    plt.figure(figsize=(12, 6))
    plt.bar(range(256), probs, width=1.0, color='darkslateblue')
    
    # Ustawienia wykresu
    output_filename = os.path.basename(plot_filename).replace('ccml_dist_', 'post_').replace('.png', '.bin')
    plt.title(f"Rozkład bajtów w {output_filename}")
    plt.xlabel("wartość bajtu (0-255)")
    plt.ylabel("prawdopodobieństwo")
    plt.xlim(-0.5, 255.5)
    plt.ylim(0, probs.max() * 1.1 if probs.max() > 0 else 0.1)
    
    # Oblicz i wyświetl entropię
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    global last_entropy
    last_entropy = entropy
    print(f"Shannon entropy ({output_filename}): {entropy:.4f} bits/symbol")
    
    # Zapisz wykres
    if plot_filename:
        plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
        print(f"Wykres zapisany → {plot_filename}")
    
    plt.close()
    
    return counts

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Użycie: ccml.py <plik_wejściowy> <plik_wyjściowy> [liczba_bitów]")
        print("  <plik_wejściowy>: Ścieżka do pliku z surowymi losowymi bitami")
        print("  <plik_wyjściowy>: Ścieżka do pliku wyjściowego")
        print("  [liczba_bitów]: Opcjonalnie - liczba bitów do wygenerowania (domyślnie 1M)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    n_bits = 1000000  # Domyślnie 1M bitów
    if len(sys.argv) > 3:
        try:
            n_bits = int(sys.argv[3])
        except ValueError:
            print(f"BŁĄD: Nieprawidłowa liczba bitów: {sys.argv[3]}")
            sys.exit(1)
    
    # Generuj wykres jeśli podano 4 argument (ścieżkę do pliku z wykresem)
    plot_file = None
    if len(sys.argv) > 4:
        plot_file = sys.argv[4]
    
    run_ccml(input_file, output_file, n_bits, plot_file)
