import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import sys
import os

# ---------------------------------------------------------------------
# 1. Chaotic tent map & CCML lattice
# ---------------------------------------------------------------------
ALPHA = 1.99999   
last_entropy = 0.0  # Globalna zmienna do przechowywania ostatniej entropii

def piecewise_map(x: float, alpha: float = ALPHA) -> float:
    #"""Mapowanie kawałkowe (tent map) z parametrem α."""
    return alpha * x if x <= 0.5 else alpha * (1.0 - x)

def ccml_step(states: np.ndarray,
              alpha: float = ALPHA,
              epsilon: float = 0.05) -> np.ndarray:
    
    L = states.size
    new_states = np.empty_like(states)
    for i in range(L):
        left   = piecewise_map(states[(i - 1) % L], alpha)
        center = piecewise_map(states[i],             alpha)
        right  = piecewise_map(states[(i + 1) % L], alpha)
        new_states[i] = ((1.0 - epsilon) * center +
                         (epsilon * 0.5) * (left + right))
    return new_states


# ---------------------------------------------------------------------
# 2. Operacje bitowe
# ---------------------------------------------------------------------
def bit_swap64(val: np.uint64, L_half: int = 32) -> np.uint64:
    # Zamiast: mask = (np.uint64(1) << L_half) - np.uint64(1)
    # Użyj:
    mask = np.left_shift(np.uint64(1), L_half) - np.uint64(1)
    
    left  = np.right_shift(val, L_half) & mask
    right = val & mask
    return np.bitwise_or(np.left_shift(right, L_half), left)


# ---------------------------------------------------------------------
# 3. Inicjalizacja CCML
# ---------------------------------------------------------------------
def initialize_states(L: int) -> np.ndarray:
    
    ps_values = [
        0.1415926535, 0.6535897932, 0.7932384626, 0.4626433832,
        0.3832795028, 0.5028841971, 0.1971693993, 0.3993751058 # L=8 values
    ]
    # Fallback values if L > 8, ensuring variety
    base_fallback = [
        0.2718281828, 0.1732050808, 0.7071067812, 0.5772156649,
        0.6931471806, 0.3010299957, 0.4142135624, 0.1234567890,
        0.9876543210, 0.3141592653, 0.8765432109, 0.5432109876
    ]
    
    idx = 0
    while len(ps_values) < L:
        ps_values.append(base_fallback[idx % len(base_fallback)])
        idx +=1
        
    return np.array(ps_values[:L], dtype=np.float64)


# ---------------------------------------------------------------------
# 4. We/Wy bitów
# ---------------------------------------------------------------------
def read_bits_from_binfile(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        byte_data = np.frombuffer(f.read(), dtype=np.uint8)
    return np.unpackbits(byte_data)                       # MSB-first

def save_bits_to_binfile(bits: np.ndarray, output_filename: str) -> None:
    bits = bits.ravel() # Upewnij się, że tablica jest 1D
    padded_len = ((bits.size + 7) // 8) * 8
    packed = np.packbits(np.pad(bits, (0, padded_len - bits.size),
                                constant_values=0))
    with open(output_filename, "wb") as f:
        f.write(packed)


# ---------------------------------------------------------------------
# 5. Główna procedura
# ---------------------------------------------------------------------
def run_ccml(filename: str = "source.bin",
             output_filename: str = "post.bin",
             N_target_bits: int = 13_000_000,
             L: int = 8,
             alpha: float = ALPHA,
             epsilon: float = 0.05,
             omega: float = 0.5,
             b_perturb: int = 3,
             word_bits: int = 64, 
             plot_filename: str | None = None,
             verbose: bool = True) -> None:

    input_bits = read_bits_from_binfile(filename)
    gamma = L // 2
    states = initialize_states(L)
    output_bits = np.empty(N_target_bits, dtype=np.uint8)
    out_pos = 0
    in_pos = 0
    MOD64 = np.uint64(0xFFFFFFFFFFFFFFFF)

    while out_pos < N_target_bits:
        needed = L * b_perturb
        if in_pos + needed > input_bits.size:
            if verbose:
                print(f"Ostrzeżenie: Brak wystarczającej liczby bitów wejściowych ({input_bits.size - in_pos}) dla pełnej rundy perturbacji (potrzebne: {needed}). Zakończenie.")
            break
        
        current_states_before_perturb = np.copy(states)
        for j in range(L):
            yc_bits = input_bits[in_pos : in_pos + b_perturb]
            in_pos += b_perturb
            yc_int = int("".join(map(str, yc_bits.tolist())), 2)
            term_yc_denominator = (2**b_perturb - 1)
            if term_yc_denominator == 0: term_yc_denominator = 1
            term_yc = omega * yc_int / term_yc_denominator
            states[j] = (term_yc + current_states_before_perturb[j]) / (1.0 + omega)
            states[j] = np.clip(states[j], 0.0, 1.0)

        for _ in range(gamma):
            states = ccml_step(states, alpha, epsilon)

        # Create a correct view of states as uint64 array
        z = np.zeros(L, dtype=np.uint64)
        for k in range(L):
            # Convert float values to uint64 by scaling and casting
            z[k] = np.uint64(states[k] * (2**64 - 1))

        for j in range(L // 2):
            idx_b = j + (L // 2)
            swap_val = bit_swap64(z[idx_b], word_bits // 2)
            # Change from addition to XOR: z_i = z_i ⊕ swap(z_{i+L/2})
            z[j] = np.uint64(np.uint64(z[j]) ^ np.uint64(swap_val))

        for j in range(L // 2):
            if out_pos >= N_target_bits:
                break
                
            # Convert z[j] to bytes properly
            z_bytes = z[j].tobytes()
            if sys.byteorder == 'little':
                # For little-endian, swap the byte order
                z_bytes = bytes(reversed(z_bytes))
                
            # Convert bytes to numpy array of uint8 before unpacking bits
            bits_from_z_word = np.unpackbits(np.frombuffer(z_bytes, dtype=np.uint8))
            
            take = min(word_bits, N_target_bits - out_pos)
            output_bits[out_pos : out_pos + take] = bits_from_z_word[:take]
            out_pos += take
    
    final_output_bits = output_bits[:out_pos]
    if plot_filename:
        if final_output_bits.size > 0:
            num_total_bits = final_output_bits.size
            padded_len_for_stats = ((num_total_bits + 7) // 8) * 8
            padded_bits_for_stats = np.pad(final_output_bits, (0, padded_len_for_stats - num_total_bits), constant_values=0)
            byte_array_for_stats = np.packbits(padded_bits_for_stats)

            if byte_array_for_stats.size > 0:
                counts = Counter(byte_array_for_stats)
                probs = np.array([counts.get(i, 0) / byte_array_for_stats.size for i in range(256)])
                plt.figure(figsize=(12, 6))
                plt.bar(range(256), probs, width=1.0, color='darkslateblue')
                plt.title(f"Rozkład bajtów w {os.path.basename(output_filename)}")
                plt.xlabel("wartość bajtu (0-255)")
                plt.ylabel("prawdopodobieństwo")
                plt.xlim(-0.5, 255.5)
                plt.ylim(0, probs.max() * 1.1 if probs.max() > 0 else 0.1)
                plt.grid(True, alpha=0.3)
                
                entropy = -sum(p * np.log2(p) for p in probs if p > 0)
                global last_entropy
                last_entropy = entropy
                if verbose:
                    print(f"Shannon entropy ({os.path.basename(output_filename)}): {entropy:.4f} bits/symbol")
                plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
                if verbose:
                    print(f"Wykres zapisany → {plot_filename}")
                plt.close()
            else:
                if verbose:
                    print("Brak danych bajtowych do wygenerowania wykresu.")
        else:
            if verbose:
                print("Brak bitów wyjściowych do wygenerowania wykresu.")

    save_bits_to_binfile(final_output_bits, output_filename)
    if verbose:
        print(f"{output_filename} zapisany ({final_output_bits.size} bitów).")
    
    return final_output_bits

# ---------------------------------------------------------------------
# 6. Uruchamianie ze skryptu
# ---------------------------------------------------------------------
if __name__ == "__main__":
    in_file  = sys.argv[1] if len(sys.argv) > 1 else "source.bin"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "post.bin"
    
    plot_dir = os.path.dirname(out_file) or "." # Użyj bieżącego katalogu, jeśli dirname jest pusty
    plot_basename = os.path.splitext(os.path.basename(out_file))[0] + "_hist.png"
    plot_png = os.path.join(plot_dir, plot_basename)

    print("== CCML post-processing ==")
    print(f"  wejście : {in_file}")
    print(f"  wyjście : {out_file}")
    print(f"  wykres  : {plot_png}")
    
    N_TARGET = 13_000_000 
    if len(sys.argv) > 3:
        try:
            N_TARGET = int(sys.argv[3])
        except ValueError:
            print(f"Ostrzeżenie: Trzeci argument ('{sys.argv[3]}') nie jest poprawną liczbą. Używam domyślnej liczby bitów: {N_TARGET}")
    print(f"  N bitów : {N_TARGET}")

    run_ccml(filename=in_file,
             output_filename=out_file,
             N_target_bits=N_TARGET,
             L=8,
             alpha=ALPHA,
             epsilon=0.05,
             omega=0.5,
             b_perturb=3,
             word_bits=64,
             plot_filename=plot_png,
             verbose=True)
    print("== CCML zakończone ==")

