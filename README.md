# TRNG - Generator Prawdziwie Losowych Liczb

## Opis Ogólny

Projekt implementuje system generowania prawdziwie losowych liczb (True Random Number Generator) z wykorzystaniem szumu audio oraz post-processingu opartego na algorytmie Coupled Chaotic Map Lattice (CCML). System pozwala na przetwarzanie istniejących plików audio jak również ciągłe nagrywanie i przetwarzanie w czasie rzeczywistym.

## Struktura Projektu

```
TRNG/
├── data/           # Katalog na pliki wejściowe i wyjściowe w trybie przetwarzania wsadowego
├── docs/           # Dokumentacja projektu
├── examples/       # Przykłady użycia
├── output/         # Katalog na wyjściowe pliki generowane w trybie ciągłym
├── scripts/        # Skrypty pomocnicze
├── src/            # Kod źródłowy projektu
│   ├── ccml.py                    # Implementacja algorytmu CCML
│   ├── ccml_optimized.py          # Zoptymalizowana wersja CCML z użyciem Numba
│   ├── pre_procesing.py           # Przetwarzanie wsadowe plików audio
│   └── realtime_trng_continuous.py # Ciągłe nagrywanie i przetwarzanie w czasie rzeczywistym
└── tests/          # Testy
```

## Tryby Działania

System oferuje dwa główne tryby działania:

### 1. Przetwarzanie Wsadowe (`pre_procesing.py`)

Skrypt `pre_procesing.py` wykonuje przetwarzanie wsadowe istniejących plików audio:

1. Wczytuje surowe próbki audio z pliku wejściowego (domyślnie `data/raw_audio_u8_44k.u8`).
2. Wybiera określony fragment danych na podstawie zdefiniowanego offsetu i wymaganej liczby bitów.
3. Ekstrahuje 3 najmniej znaczące bity (LSB) z każdej próbki audio, gdzie występuje największa entropia.
4. Konwertuje sekwencję 3 LSB na ciągły strumień bitów.
5. Pakuje strumień bitów w bajty i zapisuje do pliku `data/source.bin`.
7. Generuje i zapisuje wykresy wizualizujące rozkłady danych.
8. Oblicza i wyświetla entropię Shannona dla wartości 3 LSB.
9. Automatycznie wywołuje algorytm post-processingu CCML.

### 2. Ciągłe Przetwarzanie w Czasie Rzeczywistym (`realtime_trng_continuous.py`)

Skrypt `realtime_trng_continuous.py` umożliwia ciągłe nagrywanie dźwięku z mikrofonu i generowanie losowych danych:

1. Wyświetla listę dostępnych urządzeń wejściowych audio i pozwala na wybór preferowanego urządzenia.
2. Systematycznie nagrywa fragmenty dźwięku (domyślnie po 5 sekund).
3. Ekstrahuje 3 najmniej znaczące bity z każdej próbki dźwiękowej i gromadzi je w buforze.
4. Co określony czas (domyślnie 120 sekund) przetwarza zebrane dane, generując pliki z losowymi bitami.
5. Wywołuje algorytm CCML do post-processingu danych.
6. Generuje wykresy i statystyki na temat jakości danych.
7. System został zoptymalizowany pod kątem zarządzania pamięcią i kontroli bufora, aby zapewnić stabilne długotrwałe działanie.

### Post-processing CCML (`ccml.py` i `ccml_optimized.py`)

Moduły CCML realizują post-processing danych losowych:

1. `ccml.py` - standardowa implementacja algorytmu Coupled Chaotic Map Lattice.
2. `ccml_optimized.py` - wersja zoptymalizowana z użyciem Numba dla lepszej wydajności.

Proces przetwarzania obejmuje:

1. Inicjalizację stanów sieci CCML z wykorzystaniem bitów wejściowych jako źródła entropii.
2. Iteracyjne przetwarzanie stanów sieci z zastosowaniem chaotycznego odwzorowania namiotowego (tent map).
3. Stosowanie sprzężeń między komórkami w sieci CCML.
4. Konwertowanie stanów sieci na strumień bitów o wysokiej jakości statystycznej.
5. Zapisywanie wygenerowanych danych do pliku wyjściowego.

## Jak Uruchomić

### Tryb wsadowy (przetwarzanie istniejących plików)

1. Upewnij się, że plik z danymi audio (np. `raw_audio_u8_44k.u8`) znajduje się w katalogu `data/`.
2. Uruchom skrypt przetwarzania wsadowego:
   ```bash
   cd src
   python pre_procesing.py
   ```
3. Wyniki zostaną zapisane w katalogu `data/`.

### Tryb ciągłego nagrywania w czasie rzeczywistym

1. Uruchom skrypt ciągłego nagrywania:
   ```bash
   cd src
   python realtime_trng_continuous.py
   ```
2. Postępuj zgodnie z instrukcjami w konsoli:
   - Wybierz urządzenie wejściowe audio (lub użyj domyślnego)
   - Program będzie automatycznie zbierać dane audio i generować pliki losowe
   - Naciśnij Ctrl+C, aby zakończyć nagrywanie
3. Wyniki zostaną zapisane w katalogu `output/`, z nazwami zawierającymi znaczniki czasowe.

### Samodzielne użycie algorytmu CCML

Algorytm CCML można uruchomić niezależnie na własnym pliku z bitami:

```bash
cd src
python ccml.py <plik_wejsciowy> <plik_wyjsciowy> [liczba_bitów]
```

Przykład:
```bash
python ccml.py ../data/source.bin ../data/my_random_data.bin 13000000
```

Dla wersji zoptymalizowanej (szybsza):
```bash
python ccml_optimized.py ../data/source.bin ../data/my_random_data.bin 13000000
```

## Generowane Pliki

### W trybie wsadowym (katalog `data/`)

* `source.bin`: Strumień bitów po wstępnym przetworzeniu
* `raw_audio_histogram.png`, `raw_audio_distribution.png`: Wykresy danych wejściowych
* `post.bin`: Finalny strumień bitów po przetworzeniu przez CCML
* `ccml_post_bin_distribution.png`: Rozkład prawdopodobieństwa wyjściowych danych

### W trybie ciągłym (katalog `output/`)

* `source_HHMMSS.bin`: Strumień bitów wejściowych
* `post_HHMMSS.bin`: Strumień bitów po przetworzeniu przez CCML
* `raw_hist_HHMMSS.png`, `raw_dist_HHMMSS.png`, `lsb3_hist_HHMMSS.png`: Wykresy dla danych wejściowych
* `ccml_dist_HHMMSS.png`: Rozkład prawdopodobieństwa danych wyjściowych

## Zależności

Do uruchomienia projektu wymagane są następujące biblioteki Python:

* `numpy` - obsługa obliczeń numerycznych
* `matplotlib` - generowanie wykresów i wizualizacji
* `pyaudio` - obsługa nagrywania dźwięku w czasie rzeczywistym
* `numba` - opcjonalnie, do optymalizacji wydajności (dla `ccml_optimized.py`)

Instalacja zależności:
```bash
# Podstawowe zależności
pip install numpy matplotlib

# Do nagrywania w czasie rzeczywistym
pip install pyaudio

# Do wersji zoptymalizowanej
pip install numba
```

Na systemach Linux przed instalacją PyAudio może być konieczne zainstalowanie biblioteki portaudio:
```bash
# Debian/Ubuntu
sudo apt-get install portaudio19-dev

# Fedora/RHEL/CentOS
sudo dnf install portaudio-devel
```

## Parametry Konfiguracyjne

### Tryb ciągły (`realtime_trng_continuous.py`)

* `CHUNK` - rozmiar paczki audio (domyślnie 4096 próbek)
* `RATE` - częstotliwość próbkowania (domyślnie 44.1 kHz)
* `N_BITS_PER_CHUNK` - liczba bitów generowanych w jednej iteracji (domyślnie 13M)
* `INTERVAL_SECONDS` - czas między generowaniem plików (domyślnie 120s)
* `SAMPLE_TIME` - czas nagrywania w każdym cyklu (domyślnie 5s)
* `MAX_BUFFER_MULTIPLIER` - kontrola rozmiaru bufora

### CCML

* `ALPHA` - parametr chaotyczny (domyślnie 1.99999)
* `EPSILON` - parametr sprzężenia (domyślnie 0.05)
* `L` - rozmiar sieci CCML (domyślnie 128 lub 8, zależnie od implementacji)


Projekt TRNG oparty na CCML został zainspirowany pracą Teh, J. S., Samsudin, A., & Al-Mazrooie, M. (2020) dotyczącą post-processingu generatorów liczb losowych przy użyciu chaotycznych sieci sprzężonych.

## Licencja

[MIT License](LICENSE)
