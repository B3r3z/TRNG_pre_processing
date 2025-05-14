# Generator Liczb Losowych (TRNG) oparty na CCML

## Opis Ogólny

Projekt ten implementuje dwuetapowy proces generowania danych losowych. Pierwszy etap (`pre_procesing.py`) polega na ekstrakcji bitów z surowych próbek audio. Drugi etap (`ccml.py`) wykorzystuje te bity jako źródło entropii dla algorytmu Coupled Chaotic Map Lattice (CCML) w celu dalszego post-processingu i poprawy jakości danych losowych, zgodnie z podejściem opisanym w Teh et al. (2020).

## Krok 1: Przetwarzanie Wstępne (`pre_procesing.py`)

Skrypt `pre_procesing.py` wykonuje następujące zadania:

1.  Wczytuje surowe próbki audio z pliku `.u8` (domyślnie `raw_audio_u8_44k.u8`).
2.  Wybiera określony fragment danych na podstawie zdefiniowanego offsetu i wymaganej liczby bitów.
3.  Ekstrahuje 3 najmniej znaczące bity (LSB) z każdej próbki audio.
4.  Zapisuje wartości 3 LSB do pliku `source.u8`.
5.  Konwertuje sekwencję 3 LSB na ciągły strumień bitów.
6.  Pakuje strumień bitów w bajty i zapisuje do pliku `source.bin`. Ten plik stanowi wejście dla skryptu `ccml.py`.
7.  Generuje i zapisuje do plików PNG następujące wizualizacje:
    *   Histogram surowych próbek audio (`raw_audio_histogram.png`).
    *   Rozkład prawdopodobieństwa surowych próbek audio (`raw_audio_distribution.png`).
8.  Oblicza i wyświetla entropię Shannona dla wartości 3 LSB.
9.  **Automatycznie wywołuje skrypt `ccml.py`** w celu przetworzenia wygenerowanego pliku `source.bin`.

## Krok 2: Post-processing CCML (`ccml.py`)

Skrypt `ccml.py` jest wywoływany przez `pre_procesing.py` i realizuje następujące operacje:

1.  Wczytuje strumień bitów z pliku `source.bin`.
2.  Inicjalizuje stany sieci CCML (Coupled Chaotic Map Lattice).
3.  Iteracyjnie przetwarza stany sieci, wykorzystując bity wejściowe do perturbacji systemu.
4.  Stosuje mapowanie chaotyczne (tent map) i sprzężenie międzykomórkowe.
5.  Konwertuje stany sieci na 64-bitowe liczby całkowite, stosując reinterpretację bitów (IEEE-754 float -> uint64).
6.  Wykonuje operacje mieszania bitów, w tym zamianę połówek 64-bitowych słów (`bit_swap64`) oraz dodawanie modularne (mod 2<sup>64</sup>).
7.  Ekstrahuje przetworzone bity i zapisuje je do pliku wyjściowego `post.bin`.
8.  Generuje i zapisuje do pliku PNG rozkład prawdopodobieństwa bajtów w wynikowym pliku `post.bin` (domyślnie `ccml_post_bin_distribution.png`).
9.  Oblicza i wyświetla entropię Shannona dla przetworzonych bajtów.

## Jak Uruchomić

Aby uruchomić cały proces generowania danych losowych:

1.  Upewnij się, że plik z surowymi danymi audio (np. `raw_audio_u8_44k.u8`) znajduje się w tym samym katalogu co skrypty.
2.  Uruchom skrypt `pre_procesing.py` z linii poleceń:
    ```bash
    python pre_procesing.py
    ```
    Skrypt ten automatycznie przetworzy dane audio, zapisze plik `source.bin` oraz niezbędne wykresy, a następnie wywoła `ccml.py`, który wygeneruje `post.bin` i jego wykres dystrybucji.

    Opcjonalnie, skrypt `ccml.py` można uruchomić manualnie (np. w celu przetworzenia istniejącego pliku `source.bin` z innymi parametrami):
    ```bash
    python ccml.py <plik_wejsciowy> <plik_wyjsciowy> [liczba_bitow]
    ```
    Przykład:
    ```bash
    python ccml.py source.bin my_random_data.bin 1000000
    ```

## Generowane Pliki

Po pomyślnym wykonaniu skryptu `pre_procesing.py` (który wywołuje `ccml.py`), w katalogu projektu pojawią się następujące pliki:

*   `source.u8`: Surowe wartości 3 LSB.
*   `source.bin`: Strumień bitów po wstępnym przetworzeniu (wejście dla CCML).
*   `raw_audio_histogram.png`: Histogram surowych próbek audio.
*   `raw_audio_distribution.png`: Rozkład prawdopodobieństwa surowych próbek audio.
*   `post.bin`: Finalny strumień bitów po przetworzeniu przez CCML.
*   `ccml_post_bin_distribution.png`: Rozkład prawdopodobieństwa bajtów w pliku `post.bin`.

## Zależności

Do uruchomienia skryptów wymagane są następujące biblioteki Python:

*   `numpy`
*   `matplotlib`

Można je zainstalować za pomocą pip:
```bash
pip install numpy matplotlib
```