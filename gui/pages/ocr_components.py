import cv2
import easyocr
import numpy as np
import os

class ComponentProcessing:
    def __init__(self, input_root, preprocessed_output_root, output_best_txt, output_all_txt):
        self.reader = easyocr.Reader(['en'])
        
        self.input_root = input_root  # np. "output_components"
        self.preprocessed_output_root = preprocessed_output_root  # np. "preprocessed_rot_components"
        self.output_best_txt = output_best_txt
        self.output_all_txt = output_all_txt
        os.makedirs(self.preprocessed_output_root, exist_ok=True)

    def rotate_image(self, image, angle):
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        raise ValueError("Nieprawidłowy kąt obrotu")

    def preprocess_image(self, image):
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        sharp = cv2.filter2D(denoised, -1, sharpen_kernel)
        thresh = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        return thresh

    def is_valid_text(self, text):
        return all(char.isupper() or char.isdigit() for char in text)

    def perform_ocr_on_crop(self, crop_img, save_dir, base_name):
        best_text = ""
        best_angle = None
        all_results = []

        for angle in [0, 90, 180, 270]:
            rotated = self.rotate_image(crop_img, angle)
            preprocessed = self.preprocess_image(rotated)

            # Zapisz obraz po preprocessingu
            out_filename = f"{base_name}_rot{angle}.png"
            cv2.imwrite(os.path.join(save_dir, out_filename), preprocessed)

            results = self.reader.readtext(preprocessed)
            detected_text = " ".join([res[1] for res in results]).strip()

            all_results.append((angle, detected_text))

            if detected_text and self.is_valid_text(detected_text) and len(detected_text) >= 3:
                if len(detected_text) > len(best_text):
                    best_text = detected_text
                    best_angle = angle

        return best_angle, best_text, all_results

    def process_components(self):
        print(f"\n🔍 Przeglądam folder: {self.input_root}")
        print(f"Zawartość folderu: {os.listdir(self.input_root)}")
        with open(self.output_best_txt, "w", encoding="utf-8") as best_file, \
             open(self.output_all_txt, "w", encoding="utf-8") as all_file:

        
                for filename in os.listdir(self.input_root):
                    filepath = os.path.join(self.input_root, filename)

                    if not os.path.isfile(filepath):
                        continue
                    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        print(f"⚠️ Pomijam plik bez obsługiwanego rozszerzenia: {filename}")
                        continue

                    print(f"➡️  Przetwarzam: {filename}")

                    image = cv2.imread(filepath)
                    if image is None:
                        print(f"❌ Nie można wczytać {filename}")
                        continue

                    base_name = os.path.splitext(filename)[0]

                    # komponent typu na podstawie prefiksu przed "_"
                    component_type = base_name.split("_")[0]

                    # Folder zapisu przetworzonych wersji
                    out_component_dir = os.path.join(self.preprocessed_output_root, component_type)
                    os.makedirs(out_component_dir, exist_ok=True)

                    best_angle, best_text, all_rotations = self.perform_ocr_on_crop(
                        image, out_component_dir, base_name
                    )

                    # Zapisz do pliku .txt
                    best_file.write(f"{filename}: [{best_angle}°] -> {best_text if best_text else '(brak tekstu)'}\n")
                    all_file.write(f"===== {filename} =====\n")
                    for angle, text in all_rotations:
                        all_file.write(f"Rotacja {angle}°: {text if text else '(brak tekstu)'}\n")
                    all_file.write("\n")

                    print(f"✅ Najlepszy obrót: {best_angle}°, OCR: \"{best_text}\"")
