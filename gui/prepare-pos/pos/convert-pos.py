import csv
import re

# Ścieżka wejściowa – plik KiCad POS
INPUT_FILE = "new-top.pos"  # zmień na właściwą ścieżkę
# Ścieżka wyjściowa – uproszczony plik CSV
OUTPUT_FILE = "simplified.pos"

def convert_kicad_pos_to_csv(input_path, output_path):
    """
    Czyta plik POS w formacie KiCad i zapisuje uproszczony CSV z nagłówkiem:
    Designator,Val,Package,PosX_mm,PosY_mm,Rotation,Side
    Pomija linie komentarzy oraz nagłówki zaczynające się od '#' lub '###'.
    Zakłada, że dane są oddzielone dowolną ilością spacji.
    """
    data = []
    # Wzorzec wyłapujący tokeny – dzieląc linię na „słowa”
    token_pattern = re.compile(r'\S+')

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Pomiń linie, które zaczynają się od '#' lub są puste
            if line.lstrip().startswith("#") or line.strip() == "":
                continue
            # Wyłuskaj tokeny
            tokens = token_pattern.findall(line)
            # Przyjmujemy, że kolumny są następujące:
            # [Ref, Val, Package, PosX, PosY, Rot, Side, ...]
            if len(tokens) < 7:
                continue
            # Wybierz pierwsze 7 tokenów (ignorujemy resztę)
            ref, val, package, posx, posy, rot, side = tokens[:7]
            # Opcjonalnie – można dokonać dodatkowych modyfikacji, np. zmienić nazwę pakietu.
            # Dla przykładu, możesz zmienić "REF**" na oczekiwany designator, jeżeli taka konwersja jest wymagana.
            # Jeśli w Twoim oczekiwanym formacie zamiast "REF**" oczekujesz np. "C1", to trzeba by zmapować wartości.
            # W tym przykładzie zakładamy, że dane w pliku KiCad zawierają już poprawne designatory.
            # Jeśli nie, możesz np.:
            #   if ref.startswith("REF"):
            #       ref = ref.replace("REF", "").strip()   lub użyć innej logiki.
            # Przykładowo pozostawiamy bez zmian:
            data.append([ref, val, package, posx, posy, rot, side.capitalize()])

    # Zapisz dane do pliku CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        # Zapisz nagłówek zgodny z oczekiwanym formatem
        writer.writerow(["Designator", "Val", "Package", "PosX_mm", "PosY_mm", "Rotation", "Side"])
        for row in data:
            writer.writerow(row)
    print(f"Uproszczony plik POS zapisany jako: {output_path}")

if __name__ == '__main__':
    convert_kicad_pos_to_csv(INPUT_FILE, OUTPUT_FILE)
