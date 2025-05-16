#!/usr/bin/env python3
# filepath: /home/bartosz/Documents/PROJECT/TRNG/ccml_optimized.py

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import sys
import os
from numba import njit, prange

# ---------------------------------------------------------------------
# 1. Chaotic tent map & CCML lattice - optymalizacja Numba
# ---------------------------------------------------------------------
ALPHA = 1.99999   

@njit
def piecewise_map(x: float, alpha: float = ALPHA) -> float:
    """Mapowanie kawałkowe (tent map) z parametrem α - zoptymalizowane z Numba."""
    return alpha * x if x <= 0.5 else alpha * (1.0 - x)

@njit
def ccml_step(states: np.ndarray, alpha: float = ALPHA, epsilon: float = 0.05) -> np.ndarray:
    """Jeden krok algorytmu CCML - zoptymalizowany z Numba."""
    L = states.size
    new_states = np.empty_like(states)
    for i in range(L):
        left   = piecewise_map(states[(i - 1) % L], alpha)
        center = piecewise_map(states[i], alpha)
        right  = piecewise_map(states[(i + 1) % L], alpha)
        new_states[i] = ((1.0 - epsilon) * center + (epsilon * 0.5) * (left + right))
    return new_states

# ---------------------------------------------------------------------
# 2. Funkcje pomocnicze
# ---------------------------------------------------------------------

def print_ascii_histogram(hist_dict, max_width=50):
    max_value = max(hist_dict.values())
    for value, count in sorted(hist_dict.items()):
        bar_length = int(count / max_value * max_width)
        print(f"{value:4}: {'#' * bar_length} ({count})")

def get_binary_string(file_path, bit_count=1000):
    """Odczytaj pierwsze bit_count bitów z pliku."""
    with open(file_path, 'rb') as f:
        data = f.read((bit_count + 7) // 8)  # Odczytaj wystarczającą liczbę bajtów
        
    binary_string = ""
    for byte in data:
        binary_string += format(byte, '08b')
    
    return binary_string[:bit_count]

def split_sequence(seq, n=2):
    """Podziel sekwencję na n-gramy."""
    return ["".join(seq[i:i+n]) for i in range(0, len(seq), n)]

@njit
def calculate_new_bits(n_chaotic_bits, n_target_bits):
    """Oblicz ile bitów "zapasowych" musimy wygenerować"""
    return n_target_bits + int(n_target_bits * 0.1)  # 10% zapas

@njit
def get_bit_from_chaotic_state(x):
    """Generuj bit z wartości zmiennoprzecinkowej stanu."""
    return 1 if x > 0.5 else 0

@njit
def vectorized_get_bit(states):
    """Wektoryzowana wersja get_bit_from_chaotic_state."""
    return np.where(states > 0.5, 1, 0)

@njit(parallel=True)
def generate_bits_from_ccml(states, n_steps, n_bits, alpha=ALPHA, epsilon=0.05):
    """
    Generowanie bitów z układu CCML z określoną liczbą kroków.
    Zoptymalizowana wersja z Numba.
    """
    bits = np.zeros(n_bits, dtype=np.int8)
    current_states = states.copy()
    bit_index = 0
    
    # Wykonaj określoną liczbę kroków CCML
    for _ in range(n_steps):
        if bit_index >= n_bits:
            break
            
        # Wykonaj krok CCML
        current_states = ccml_step(current_states, alpha, epsilon)
        
        # Generuj bity dla każdego stanu
        state_bits = vectorized_get_bit(current_states)
        
        # Dodaj bity do wyjścia
        bits_to_copy = min(len(state_bits), n_bits - bit_index)
        bits[bit_index:bit_index + bits_to_copy] = state_bits[:bits_to_copy]
        bit_index += bits_to_copy
    
    return bits

# ---------------------------------------------------------------------
# 3. Główne funkcje CCML
# ---------------------------------------------------------------------

def run_ccml(filename, output_filename, N_target_bits=1000000, plot_filename=None, verbose=True):
    """
    Główna funkcja do uruchamiania algorytmu CCML na pliku wejściowym.
    
    Parametry:
    - filename (str): Ścieżka do pliku wejściowego (źródło losowych bitów)
    - output_filename (str): Ścieżka do pliku wyjściowego
    - N_target_bits (int): Liczba bitów do wygenerowania
    - plot_filename (str, optional): Ścieżka do zapisu wykresu rozkładu
    - verbose (bool): Czy wyświetlać szczegółowe informacje
    """
    if not os.path.exists(filename):
        print(f"BŁĄD: Plik {filename} nie istnieje.")
        return
        
    # Wczytaj sekwencję bitów
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
    n_steps = (n_chaotic_bits + L - 1) // L  # Ile kroków potrzeba
    
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
    """Analizuj rozkład n-gramów bitowych."""
    n = 4  # Analizuj 4-gramy
    
    # Konwertuj bity na ciąg znaków
    bitstring = ''.join(str(bit) for bit in bits[:1000000])  # Użyj maksymalnie 1M bitów dla szybkości
    
    # Podziel na n-gramy
    ngrams = split_sequence(bitstring, n)
    
    # Policz wystąpienia
    counts = Counter(ngrams)
    
    # Wypełnij brakujące n-gramy zerami
    all_patterns = [format(i, f'0{n}b') for i in range(2**n)]
    for pattern in all_patterns:
        if pattern not in counts:
            counts[pattern] = 0
    
    # Posortuj według wartości binarnej
    sorted_counts = {k: counts[k] for k in sorted(counts.keys())}
    
    # Oblicz teoretyczny rozkład (powinien być równomierny)
    total_ngrams = len(ngrams)
    expected_count = total_ngrams / (2**n)
    
    # Rysuj histogram
    plt.figure(figsize=(12, 6))
    
    # Indeksy dla osi X (wartości dziesiętne dla każdego wzorca binarnego)
    x_indices = [int(pattern, 2) for pattern in sorted_counts.keys()]
    
    # Rysuj słupki
    plt.bar(x_indices, list(sorted_counts.values()), color='blue', alpha=0.7)
    
    # Rysuj linię oczekiwaną
    plt.axhline(y=expected_count, color='r', linestyle='-', label='Oczekiwany rozkład')
    
    plt.title(f"Rozkład {n}-gramów po CCML")
    plt.xlabel(f"{n}-gram (binarny)")
    plt.ylabel("Liczba wystąpień")
    plt.xticks([0, 5, 10, 15], ['0000', '0101', '1010', '1111'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if plot_filename:
        plt.savefig(plot_filename, dpi=300)
    
    plt.close()
    
    return sorted_counts

# ---------------------------------------------------------------------
# Uruchomienie bezpośrednie
# ---------------------------------------------------------------------

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
