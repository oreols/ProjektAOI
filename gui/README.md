# Projekt AOI - Gałąź GUI

## Wymagania
- Python 3.x
- MySQL
- PyQt5: pip install PyQt5
- bcrypt: pip install bcrypt
- mysql-connector-python: pip install mysql-connector-python
- git (opcjonalnie, do pobrania projektu)

**Instalacja projektu**
1. **Sklonowanie repozytorium:**

git clone -b gui https://github.com/oreols/ProjektAOI.git
cd ProjektAOI

2. **Import bazy danych:**
Najpierw upewnij się, że baza danych o nazwie aoi_pcb istnieje. Jeśli nie, utwórz ją: CREATE DATABASE aoi_pcb;
Następnie zaimportuj bazę z pliku dumpa: mysql -u root -p aoi_pcb < aoi_pcb_dump.sql

3. ## Uruchomienie aplikacji

1. Upewnij się, że serwer MySQL jest uruchomiony.
2. Uruchom aplikację : python main.py
3. Dane logowania do konta administratora:
email: admin@onet.pl
hasło: PassWord123.?

## Struktura projektu
- `main.py` - Plik startowy aplikacji
- `pages/` - Moduły odpowiedzialne za poszczególne ekrany (logowanie, rejestracja, menu)
- `ui/` - Pliki interfejsu graficznego stworzone w Qt Designer
- `aoi_pcb_dump.sql` - Zrzut bazy danych

## Używanie aplikacji
1. Zaloguj się na istniejące konto wtedy w Menu -> Dodaj konto możesz stworzyć nowych użytkowników.
2. W przypadku logowania jako administrator możesz rejestrować nowych użytkowników oraz nadawać im uprawnienia.

Miłego użytkowania! 

