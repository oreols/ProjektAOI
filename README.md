# Projekt AOI - Automatyczna Optyczna Inspekcja

## Spis treści
1. [Wprowadzenie](#wprowadzenie)
2. [Wymagania systemowe](#wymagania-systemowe)
3. [Instalacja](#instalacja)
4. [Struktura projektu](#struktura-projektu)
5. [Moduł ML](#moduł-ml)
6. [Interfejs GUI](#interfejs-gui)
7. [Instrukcja obsługi](#instrukcja-obsługi)
8. [Konfiguracja bazy danych](#konfiguracja-bazy-danych)
9. [Rozwiązywanie problemów](#rozwiązywanie-problemów)
10. [Rozwój projektu](#rozwój-projektu)

## Wprowadzenie

Projekt AOI (Automatyczna Optyczna Inspekcja) służy do automatycznej analizy i weryfikacji komponentów elektronicznych na płytkach PCB za pomocą technik wizji komputerowej i uczenia maszynowego. System pozwala na:

- Wykrywanie i klasyfikację różnych typów komponentów elektronicznych
- Porównywanie rzeczywistego rozmieszczenia komponentów z danymi z pliku POS
- Analizę rozmieszczenia i poprawności montażu
- Automatyczne wykrywanie potencjalnych błędów lub braków w montażu
- Archiwizację wyników inspekcji w bazie danych

Aplikacja integruje nowoczesne modele uczenia maszynowego z intuicyjnym interfejsem graficznym, co czyni ją przydatnym narzędziem dla inżynierów elektroników, kontroli jakości oraz działów produkcyjnych.

## Wymagania systemowe

### Sprzęt
- Procesor: min. Intel Core i5 lub odpowiednik AMD
- RAM: min. 8GB (zalecane 16GB)
- Karta graficzna wspierająca CUDA (dla przyspieszenia modeli ML)

### Oprogramowanie
- Windows 10/11 lub Linux (Ubuntu 18.04+)
- Python 3.8+ 
- PyQt5
- OpenCV 4.5+
- PyTorch 1.8+
- MySQL Server 8.0+
- Pakiety Python wymienione w pliku requirements.txt

## Instalacja

1. Sklonuj repozytorium:
```
git clone https://github.com/twojerepo/ProjektAOI.git
cd ProjektAOI
```

2. Utwórz i aktywuj wirtualne środowisko Python:
```
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Zainstaluj wymagane pakiety:
```
pip install -r requirements.txt
```

4. Skonfiguruj bazę danych MySQL:
   - Utwórz bazę danych o nazwie `aoi_database`
   - Zaimportuj baze danych.
   - Skonfiguruj plik `db_config.py` podając dane dostępowe do bazy

5. Pobierz pliki modeli i umieść je w folderze `gui/models/trained/`

## Struktura projektu

```
ProjektAOI/
├── gui/                   # Moduły interfejsu użytkownika
│   ├── pages/             # Poszczególne ekrany aplikacji
│   └── ocr_components.py  # Komponenty rozpoznawania tekstu
├── models/                # Modele uczenia maszynowego
│   ├── faster_rcnn.py     # Implementacja modelu Faster R-CNN
│   └── trained/           # Wytrenowane modele dla różnych komponentów
├── ui/                    # Pliki definicji interfejsu (.ui)
├── output_components/     # Katalog na wycięte komponenty
├── saved_images/          # Zapisane obrazy płytek PCB
├── db_config.py           # Konfiguracja bazy danych
├── main.py                # Główny plik uruchomieniowy
└── requirements.txt       # Lista zależności
```

## Moduł ML

### Dostępne modele
System wykorzystuje następujące modele do detekcji komponentów:
- Kondensatory
- Układy scalone
- Diody
- Złącza USB
- Rezonatory
- Rezystory
- Przyciski
- Złącza

### Architektura
Wszystkie modele bazują na architekturze Faster R-CNN z detektorem cech ResNet-50 FPN. Modele zostały wytrenowane na zbiorach danych zawierających setki zdjęć płytek PCB z oznaczonymi komponentami.

### Parametry modeli
Każdy model ma zdefiniowany próg pewności (confidence threshold), który można dostosować w kodzie:
```python
self.confidence_thresholds = {
    "Kondensator": 0.9,
    "Uklad scalony": 0.90,
    "Dioda": 0.55,
    "USB": 0.8,
    "Rezonator": 0.8,
    "Rezystor": 0.5,
    "Przycisk": 0.6,
    "Zlacze": 0.75,
}
```

### Trening własnych modeli
Aby wytrenować własne modele:
1. Przygotuj zbiór danych z oznaczonymi komponentami
2. Dostosuj parametry w skrypcie trenującym
3. Uruchom trening na GPU
4. Zapisz model w katalogu `models/trained/`

## Interfejs GUI

Aplikacja wykorzystuje interfejs graficzny oparty na PyQt5. Główne elementy interfejsu to:

1. **Panel podglądu obrazu** - wyświetla obraz z kamery lub wczytany obraz
2. **Panel kontrolny** - zawiera przyciski do sterowania analizą
3. **Lista komponentów** - wyświetla wykryte komponenty
4. **Panel wyników** - pokazuje statystyki i wyniki porównań

## Instrukcja obsługi

### Uruchomienie aplikacji
Aby uruchomić aplikację, wykonaj:
```
python main.py
```

### Analiza płytki PCB
1. **Wczytanie obrazu**:
   - Kliknij "Wczytaj zdjęcie" aby wybrać obraz płytki PCB
   - Alternatywnie użyj przycisku "Start kamery" aby wykorzystać obraz z kamery

2. **Preprocessing obrazu**:
   - Kliknij "Preprocessing" aby wykryć i wyciąć płytkę PCB z obrazu
   - System automatycznie wykryje krawędzie płytki i obróci ją do pozycji poziomej

3. **Nakładanie pozycji z pliku POS**:
   - Kliknij przycisk "POS" aby wybrać plik CSV z pozycjami komponentów
   - Wybierz tryb nakładania (punkty lub pola do porównania)
   - Jeśli plik POS dotyczy spodniej strony płytki, użyj "Lustro" do odbicia pozycji

4. **Analiza komponentów**:
   - Kliknij "Analiza" aby wykryć wybrany typ komponentu
   - Alternatywnie użyj "Analizuj wszystko" aby wykryć wszystkie typy komponentów
   - Wyniki zostaną wyświetlone na liście komponentów i na obrazie

5. **Porównanie z POS**:
   - Kliknij "Porównanie komponentów" aby porównać wykryte komponenty z pozycjami POS
   - System pokaże dopasowane i niedopasowane komponenty
   - Kliknij na element listy aby podświetlić go na obrazie

6. **Zapisywanie wyników**:
   - Kliknij "Zapisz" aby zapisać wyniki detekcji
   - Kliknij "Zapisz porównanie" aby zapisać wyniki porównania z POS
   - Wprowadź kod PCB lub zaakceptuj proponowany kod

### Funkcje dodatkowe
- **Praca z kamerą - NIEZOPTYMALIZOWANE W TEJ FAZIE PROJEKTU** - przyciski "Start kamery" i "Stop kamery"
- **Czyszczenie** - przycisk "Wyczyść" resetuje stan aplikacji
- **Zmiana widoku** - przycisk "Pokaż preprocessing/detekcję" przełącza widok
- **Odbicie lustrzane** - przycisk "Lustro" odbija pozycje z pliku POS

## Konfiguracja bazy danych

Aplikacja wykorzystuje bazę danych MySQL do przechowywania wyników. Plik `db_config.py` powinien zawierać:

```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'twoj_uzytkownik',
    'password': 'twoje_haslo',
    'database': 'aoi_database'
}
```

### Struktura bazy danych
System tworzy następujące tabele:
- `pcb_records` - rekordy analizowanych płytek PCB
- `components` - wykryte komponenty
- `aoi_pcb_ins_p_bboxs_comparisons` - wyniki porównań z pozycjami POS

## Rozwiązywanie problemów

### Problemy z wykrywaniem płytki PCB
- Upewnij się, że płytka ma wyraźny kontrast z tłem
- Użyj jasnego, jednolitego oświetlenia
- Jeśli automatyczne wykrywanie zawodzi, można użyć funkcji ręcznego przycinania

### Problemy z detekcją komponentów
- Sprawdź czy modele zostały poprawnie załadowane
- Dostosuj progi pewności w zależności od jakości obrazu
- Upewnij się, że obraz jest ostry i dobrze oświetlony

### Problemy z OCR
- System EasyOCR wymaga dobrej jakości obrazu
- Małe oznaczenia na komponentach mogą być trudne do odczytania
- Spróbuj użyć różnych kątów obrotu komponentu

## Rozwój projektu

### Planowane funkcje
- Wsparcie dla większej liczby typów komponentów
- Ulepszenie rozpoznawania tekstu na komponentach
- Zoptymalizowanie możlwiości wykrywania z wykorzystaniem kamery strumieniując obraz.

---

**Autorzy:** Karolina Szczepaniak, Łukasz Skrzypiec, Kacper Popko, Marek Pichniarczyk
**Prowadzący:** Mgr inż. Nikodem Bulanda 
**Przedmiot:** Przedswiewięcie inżynierskie 
**Data:** Maj 2025
