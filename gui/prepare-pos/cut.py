import os
import cv2
import pandas as pd
import numpy as np

# === KONFIGURACJA ===
IMAGE_PATH = "images/ACCC1.jpg"  # zdjęcie płytki
POS_PATH = "pos/test_board.pos"  # plik .pos
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === PARAMETRY FIZYCZNE PŁYTKI (mm) === NALEŻY PODAĆ RZECZYWISTE WYMIARY PŁTYKI W MM !!!
BOARD_WIDTH_MM = 102.0
BOARD_HEIGHT_MM = 53.0
CROP_SIZE_MM = 8.0  # obszar wokół komponentu do wycięcia

# === DETEKCJA I PRZYCINANIE PŁYTKI PCB ===
def extract_pcb(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Nie wykryto konturu płytki PCB – sprawdź kolor lub jakość zdjęcia.")

    pcb_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(pcb_contour)

    pcb_img = image[y:y+h, x:x+w]
    rotated = False

    if h > w:
        pcb_img = cv2.rotate(pcb_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated = True
        w, h = h, w

    return pcb_img, rotated, w, h

# === PRZETWARZANIE ===
pcb_img, rotated, pcb_w_px, pcb_h_px = extract_pcb(IMAGE_PATH)
# 💾 Zapisz wersję debugową po przycięciu i rotacji
debug_img_path = os.path.join(OUTPUT_DIR, "debug_pcb_processed.jpg")
cv2.imwrite(debug_img_path, pcb_img)
print(f"[💾] Zapisano obraz debugowy płytki: {debug_img_path}")
scale_x = pcb_w_px / BOARD_WIDTH_MM
scale_y = pcb_h_px / BOARD_HEIGHT_MM
scale = (scale_x + scale_y) / 2

print(f"[INFO] Skala: 1 mm ≈ {scale:.2f} px")
print(f"[INFO] Orientacja: {'obrócona (90°)' if rotated else 'pozioma'}")

# === Wczytaj komponenty z .pos ===
df = pd.read_csv(POS_PATH)

# === Dla każdego komponentu: wytnij i zapisz ===
for _, row in df.iterrows():
    ref = row["Designator"]
    val = row["Val"]
    x_mm = float(row["PosX_mm"])
    y_mm = float(row["PosY_mm"])

    # Jeśli tworzysz plik .pos ręcznie NA PODSTAWIE OBRAZU po obróceniu – nie przeliczaj współrzędnych!
    # Jeśli .pos pochodzi z KiCada – wtedy użyj przeliczenia:
    # if rotated:
    #     x_mm, y_mm = y_mm, BOARD_WIDTH_MM - x_mm


    x_px = int(x_mm * scale)
    y_px = int(y_mm * scale)
    crop_px = int(CROP_SIZE_MM * scale)

    x1 = max(0, x_px - crop_px // 2)
    y1 = max(0, y_px - crop_px // 2)
    x2 = min(pcb_w_px, x_px + crop_px // 2)
    y2 = min(pcb_h_px, y_px + crop_px // 2)

    crop = pcb_img[y1:y2, x1:x2]
    if crop.size == 0 or x1 >= x2 or y1 >= y2:
        print(f"[❌] Błąd przy cięciu {ref} ({x_mm:.1f}, {y_mm:.1f}) mm")
        continue

    filename = f"{ref}_{val}.png".replace("/", "_")
    filepath = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(filepath, crop)
    print(f"[✔] Zapisano komponent: {filename}")

print(f"\n[✅] Wszystkie komponenty zapisane do: {OUTPUT_DIR}/")
