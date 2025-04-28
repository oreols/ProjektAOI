# Dokumentacja interfejsu graficznego dla systemu AOI (Automated Optical Inspection)

Ten dokument opisuje strukturę i funkcjonalności modułu GUI aplikacji do automatycznej inspekcji optycznej PCB.

## Struktura projektu

Katalog `gui/` zawiera następujące elementy:

```
gui/
├── main.py                    # Punkt wejścia aplikacji
├── database.py                # Funkcje obsługi bazy danych
├── db_config.py               # Konfiguracja połączenia z bazą danych
├── models/                    # Modele uczenia maszynowego
│   └── faster_rcnn.py         # Implementacja modelu Faster R-CNN
├── pages/                     # Implementacje poszczególnych ekranów
│   ├── camera.py              # Moduł analizy obrazu z kamery/zdjęć
│   ├── login.py               # Ekran logowania
│   ├── menu.py                # Główne menu aplikacji
│   ├── register.py            # Rejestracja nowych użytkowników
│   ├── account.py             # Zarządzanie kontami użytkowników
│   ├── accountsettings.py     # Ustawienia konta
│   ├── history.py             # Historia inspekcji
│   └── reports.py             # Generowanie raportów
└── ui/                        # Pliki definicji interfejsu użytkownika (Qt Designer)
    ├── Camera.ui              # Interfejs modułu kamery
    ├── Login.ui               # Ekran logowania
    ├── Menu.ui                # Główne menu
    └── ...                    # Pozostałe pliki UI
```

## Uruchamianie aplikacji

Aby uruchomić aplikację, należy wykonać:

```bash
cd gui
python main.py
```

## Moduły funkcjonalne

### Logowanie i zarządzanie użytkownikami

System obsługuje logowanie użytkowników z różnymi poziomami uprawnień. Funkcjonalności obejmują:
- Logowanie istniejących użytkowników
- Rejestracja nowych użytkowników (tylko dla administratorów)
- Zarządzanie kontami użytkowników (edycja, usuwanie)
- Zmiana ustawień konta

### Moduł kamery i analizy obrazu

Główna funkcjonalność systemu zawarta w pliku `camera.py`. Umożliwia:
- Uruchomienie strumienia z kamery
- Wczytanie zdjęcia z dysku do analizy
- Automatyczną detekcję komponentów PCB z wykorzystaniem modeli głębokiego uczenia
- Wyświetlanie wykrytych obiektów (bounding boxy)
- Identyfikację poszczególnych komponentów
- Nagrywanie obrazu z kamery
- Przełączanie między różnymi modelami detekcji dla różnych typów komponentów

#### Jak działa moduł analizy obrazu

1. **Wczytywanie zdjęcia**:
   - Przycisk "Wczytaj zdjęcie" pozwala na wybór pliku graficznego z dysku
   - Obsługiwane formaty: PNG, JPG, JPEG, BMP

2. **Analiza obrazu**:
   - Po wczytaniu obrazu, kliknięcie przycisku "Analiza" uruchamia detekcję komponentów
   - Wykryte komponenty są oznaczane kolorowymi ramkami (bounding box)
   - Lista wykrytych komponentów jest wyświetlana w panelu po prawej stronie
   - Kliknięcie na element listy powoduje wyróżnienie odpowiedniego komponentu na obrazie

3. **Obsługa modeli**:
   - Dostępne modele dla różnych typów komponentów:
     - Kondensatory
     - Układy scalone
     - Diody
     - USB
     - Rezonatory
     - Rezystory
     - Przyciski
     - Złącza

#### Szczegółowa procedura preprocessingu obrazu

System AOI wykorzystuje zaawansowany preprocessing obrazów, który ma kluczowe znaczenie dla skuteczności detekcji komponentów. Procedura preprocessingu obejmuje następujące etapy:

1. **Korekcja optyczna dystorsji**
   - Kompensacja zniekształceń obiektywu kamery
   - Wykorzystanie macierzy kalibracji kamery i współczynników dystorsji

2. **Globalna normalizacja histogramu**
   - Rozciągnięcie histogramu obrazu do pełnego zakresu 0-255
   - Poprawa kontrastu globalnego i dynamiki obrazu

3. **Redukcja szumów**
   - Filtracja medianowa lub Gaussowska w zależności od warunków
   - Adaptacyjny filtr bilateralny (parametry d=5, sigmaColor=75, sigmaSpace=75)
   - Zachowanie krawędzi przy jednoczesnym wygładzaniu obszarów jednolitych

4. **Adaptacyjne wyrównanie kontrastu (CLAHE)**
   - Zastosowanie CLAHE na kanale L w przestrzeni kolorów LAB
   - Parametry: clipLimit=2.0, tileGridSize=(8,8)
   - Poprawa lokalnego kontrastu bez nadmiernego wzmacniania szumu

5. **Regulacja kontrastu i jasności**
   - Delikatne zwiększenie kontrastu (alpha=1.1)
   - Lekkie rozjaśnienie obrazu (beta=5)

6. **Rejestracja obrazu (opcjonalnie)**
   - Wykorzystanie punktów fiducjalnych do wyrównania obrazu
   - Transformacja afiniczna w celu skorygowania perspektywy

7. **Wstępna segmentacja**
   - Konwersja do przestrzeni kolorów HSV
   - Tworzenie masek dla różnych elementów płytki PCB:
     - Maska soldermask (zielone obszary płytki)
     - Maska padów (białe lub metaliczne elementy)
     - Maska otworów

Zastosowanie takiego wieloetapowego procesu preprocessingu znacząco poprawia skuteczność detekcji przez:
- Zwiększenie kontrastu między komponentami a tłem
- Redukcję szumów zakłócających proces detekcji
- Standaryzację warunków oświetleniowych
- Uwydatnienie cech charakterystycznych komponentów

System umożliwia również przełączanie między widokiem oryginalnego obrazu a obrazem po preprocessingu, co pozwala użytkownikowi lepiej zrozumieć proces analizy i zweryfikować poprawność detekcji.

### Generowanie raportów

Moduł raportowania umożliwia:
- Przeglądanie historii przeprowadzonych inspekcji
- Generowanie raportów z detekcji
- Eksport wyników do formatów PDF/CSV

## Technologie

- **PyQt5**: Framework interfejsu graficznego
- **OpenCV**: Przetwarzanie obrazu
- **PyTorch/Torchvision**: Modele detekcji obiektów (Faster R-CNN)
- **MySQL**: Przechowywanie danych użytkowników i raportów

## Konfiguracja bazy danych

Aplikacja wymaga skonfigurowanej bazy danych MySQL. Parametry połączenia znajdują się w pliku `db_config.py`. Do inicjalizacji struktury bazy danych można wykorzystać skrypt `aoi_pcb_dump.sql`.

## Znane problemy/ograniczenia

- Modele detekcji są zoptymalizowane dla określonych typów komponentów i mogą mieć ograniczoną skuteczność dla nietypowych układów PCB
- Przy analizie zdjęć o wysokiej rozdzielczości może wystąpić opóźnienie w przetwarzaniu
- Aplikacja wymaga zainstalowanego PyTorch z obsługą CUDA dla optymalnej wydajności na GPU

## Rozszerzanie funkcjonalności

Aby dodać obsługę nowego typu komponentu:
1. Przygotuj model detekcji (Faster R-CNN) dla nowego typu
2. Dodaj ścieżkę do modelu w słowniku `model_paths` w pliku `camera.py`
3. Dodaj nowy element do listy rozwijanej w UI i kodzie

## Autorzy

Zespół Projektu AOI 