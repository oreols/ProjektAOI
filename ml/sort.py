import os
import random
import shutil

"""
    Plik odpowiedzialny za dzielenie danych na zbiór treningowy i validacyjny.
"""

# Katalog z oryginalnymi obrazami
all_images_dir = "dataset/voc_annotations"  # Zmień na swoją ścieżkę

# Katalogi docelowe
train_dir = "dataset/voc_annotations/train_voc"
val_dir = "dataset/voc_annotations/val_voc"

# Tworzymy katalogi docelowe (jeśli jeszcze nie istnieją)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Lista wszystkich plików w katalogu z obrazami
all_files = [f for f in os.listdir(all_images_dir) if f.lower().endswith(('.xml', '.png'))]

# Mieszamy listę losowo, aby podział był przypadkowy
random.shuffle(all_files)

# Ustal proporcję podziału (np. 80% do treningu, 20% do walidacji)
train_ratio = 0.9
train_count = int(len(all_files) * train_ratio)  # liczba obrazów w treningu

train_files = all_files[:train_count]
val_files = all_files[train_count:]

print(f"Całkowita liczba plików: {len(all_files)}")
print(f"Trening: {len(train_files)}")
print(f"Walidacja: {len(val_files)}")

# Kopiowanie plików do katalogów docelowych
for f in train_files:
    shutil.copy(os.path.join(all_images_dir, f), os.path.join(train_dir, f))

for f in val_files:
    shutil.copy(os.path.join(all_images_dir, f), os.path.join(val_dir, f))

print("Podział zakończony!")
