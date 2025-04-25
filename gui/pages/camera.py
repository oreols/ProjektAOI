from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
import cv2
import os
from PyQt5.QtWidgets import QDialog, QLabel, QMessageBox, QFileDialog, QListWidget, QPushButton
import torchvision
import torch
from PyQt5.QtCore import QTimer
from models.faster_rcnn import get_model
import urllib.request
import numpy as np
import sys
import matplotlib.pyplot as plt
from io import BytesIO

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Camera(QDialog):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loadUi("ui/Camera.ui", self)
        self.setGeometry(100, 100, 1200, 800)
        self.cap_label = self.findChild(QLabel, "cap")
        self.bboxes = []  # Inicjalizacja listy wykrytych obiektów
        self.component_list.itemClicked.connect(self.highlight_bbox)

        self.frozen = False  # Dodaj tę linię
        self.frozen_frame = None  # Przechowa zamrożoną klatkę
        self.original_frame = None  # Przechowa oryginalną kopię obrazu bez boxów
        self.preprocessed_frame = None  # Przechowa obraz po preprocessingu
        self.is_preprocessed = False  # Flaga czy obraz został już przetworzony
        self.pcb_contour = None  # Przechowuje kontur płytki PCB
        self.pcb_corners = None  # Przechowuje rogi płytki PCB

        self.model_paths = {
            "Kondensator": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/capacitors_model_epoch_19_mAP_0.815.pth")),
            "Układ scalony": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/ic_model_epoch_18_mAP_0.875.pth")),
            "Dioda": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/dioda_model_epoch_14_mAP_0.822.pth")),
            "USB": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/usb_model_epoch_12_mAP_0.934.pth")),
            "Rezonator": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/rezonator_model_epoch_6_mAP_0.934.pth")),
            "Rezystor": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/rezystor_model_epoch_8_mAP_0.825.pth")),
            "Przycisk": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/switch_best_model_epoch_14_mAP_0.966.pth")),
            "Złącze": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/connectors_model_epoch_10_mAP_0.733.pth")),
        }

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    
        num_classes = 2
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        
        self.model.eval()

        self.component.currentTextChanged.connect(self.change_model)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.analyze = False
        self.recording = False
        self.video_writer = None
        self.frame_count = 0

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.analyze_button.clicked.connect(self.toggle_analysis)
        self.record_button.clicked.connect(self.toggle_recording)
        self.virtual_cam_button.clicked.connect(self.choose_virtual_camera)
        self.clear_image_button.clicked.connect(self.clear_image)
        
        # Zmiana tekstu przycisku aby odzwierciedlał jego nową funkcję
        self.virtual_cam_button.setText("Wczytaj zdjęcie")
        
        # Dodajemy wszystkie modele z model_paths do listy rozwijanej
        self.component.clear()
        for component_name in sorted(self.model_paths.keys()):
            self.component.addItem(component_name)
        
        # Załaduj pierwszy model jeśli lista nie jest pusta
        if self.component.count() > 0:
            first_component = self.component.itemText(0)
            self.change_model(first_component)

        # Inicjujemy przyciski preprocessingu
        self.preprocessing_btn = self.findChild(QPushButton, "preprocessing_btn")
        self.show_preprocessing_btn = self.findChild(QPushButton, "show_preprocessing_btn")
        
        # Podłączamy sygnały do przycisków
        self.preprocessing_btn.clicked.connect(self.run_preprocessing)
        self.show_preprocessing_btn.clicked.connect(self.toggle_preprocessing_view)
        self.preprocessing_visible = False
        
        # Dezaktywuj przycisk analizy - najpierw musi być preprocessing
        self.analyze_button.setEnabled(False)
        self.show_preprocessing_btn.setEnabled(False)  # Początkowo nie ma nic do pokazania

    def load_model(self, model_path):
        if os.path.exists(model_path):
            try:
                # Użyj tej samej funkcji co w test.py
                self.model = get_model(2)
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)  # Upewnij się że model jest na właściwym urządzeniu
                self.model.eval()
                print(f"\nZaładowano model: {model_path}")
                print(f"Model device: {next(self.model.parameters()).device}")
            except RuntimeError as e:
                print("\n❌ Błąd podczas ładowania state_dict:")
                print(e)
                QMessageBox.critical(self, "Błąd", f"Problem z załadowaniem modelu:\n{e}")
                return
        else:
            QMessageBox.critical(self, "Błąd", f"Nie znaleziono modelu: {model_path}")


    def change_model(self, selected_component):
        if selected_component in self.model_paths:
            self.load_model(self.model_paths[selected_component])
            print(f"Załadowano model: {selected_component}")
        else:
            QMessageBox.warning(self, "Uwaga", f"Nie znaleziono modelu dla: {selected_component}")

    def choose_virtual_camera(self):
        # Zatrzymaj wszystkie aktywne strumienie wideo
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        self.timer.stop()
        
        # Wyczyść stare dane
        self.frozen = False
        self.frozen_frame = None
        self.original_frame = None
        self.preprocessed_frame = None
        self.is_preprocessed = False
        self.bboxes = []
        self.component_list.clear()
        self.count_elements.setText("")
        
        # Wyświetl dialog wyboru pliku
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Wybierz zdjęcie", "", 
            "Pliki obrazów (*.png *.jpg *.jpeg *.bmp);;Wszystkie pliki (*)", 
            options=options
        )
        
        if not file_name:
            # Użytkownik anulował wybór pliku
            return
            
        # Wczytaj wybrany obraz
        image = cv2.imread(file_name)
        if image is None:
            QMessageBox.critical(self, "Błąd", f"Nie udało się wczytać obrazu: {file_name}")
            return
            
        # Przechowaj obraz jako "zamrożoną klatkę" i oryginał
        self.original_frame = image.copy()  # Zachowaj kopię oryginalnego obrazu
        self.frozen_frame = image.copy()
        self.frozen = True
        print(f"Załadowano obraz: {file_name}, wymiary: {image.shape}")
        
        # Wyświetl obraz w interfejsie
        self.show_frame(image)
        
        # Aktywuj przycisk preprocessingu, dezaktywuj analizę
        self.preprocessing_btn.setEnabled(True)
        self.analyze_button.setEnabled(False)
        
        # Zaktualizuj stan interfejsu
        QMessageBox.information(self, "Informacja", "Obraz został wczytany. Teraz wykonaj preprocessing.")

    def resize_with_aspect_ratio(self, frame, target_width, target_height):
        h, w, _ = frame.shape
        aspect_ratio = w / h
        new_width = target_width
        new_height = int(new_width / aspect_ratio)

        if new_height > target_height:
            new_height = target_height
            new_width = int(new_height * aspect_ratio)

        # frame = cv2.resize(frame, (self.cap_label.width(), self.cap_label.height()), interpolation=cv2.INTER_LINEAR)
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        result = cv2.copyMakeBorder(
            resized_frame,
            (target_height - new_height) // 2,
            (target_height - new_height + 1) // 2,
            (target_width - new_width) // 2,
            (target_width - new_width + 1) // 2,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        return result

    def start_camera(self):
        if self.cap is not None and self.cap.isOpened():
            QMessageBox.warning(self, "Uwaga", "Kamera już jest włączona!")
            return

        # self.cap = cv2.VideoCapture(0) # kamera laptop
        url = "http://192.168.1.12:4747/video"
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)  #kamera telefonu
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Błąd", "Nie można uzyskać dostępu do kamery.")
            self.cap = None
            return

        # Aktywuj analizę kamery - preprocessing będzie wykonywany automatycznie
        self.preprocessing_btn.setEnabled(False)  # Dla kamery preprocessing jest automatyczny
        self.analyze_button.setEnabled(True)      # Analiza może być wykonana
        
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.cap_label.setPixmap(QPixmap())
        if self.recording:
            self.toggle_recording()

    def toggle_analysis(self):
        # Sprawdź czy preprocessing został wykonany
        if self.frozen and not self.is_preprocessed:
            QMessageBox.warning(self, "Uwaga", "Najpierw wykonaj preprocessing!")
            return
            
        self.analyze = not self.analyze
        self.analyze_button.setText("Wyłącz Analizę" if self.analyze else "Analiza")
        
        # Jeśli włączono analizę i mamy załadowany obraz statyczny
        if self.analyze and self.frozen and self.preprocessed_frame is not None and self.cap is None:
            print("Analizuję statyczny obraz...")
            
            # Użyj przetworzonego obrazu do analizy
            analyze_frame = self.preprocessed_frame.copy()
                
            print(f"Analizuję obraz o wymiarach: {analyze_frame.shape}")
                
            # Wykonaj analizę na obrazie
            result_frame, detections = self.detect_components(analyze_frame)
            self.update_component_list(detections)
            
            # Rysowanie bounding boxów na obrazie
            display_frame = self.preprocessed_frame.copy()  # Rysuj na przetworzonym obrazie
            for bbox in self.bboxes:
                x1, y1, x2, y2 = bbox["bbox"]
                color = bbox["color"]
                # Zwiększono grubość ramki z 2 na 4
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 4)
                # Dodaj etykietę z ID - zwiększono wielkość fonta z 0.5 na 1.0 i grubość z 2 na 3
                cv2.putText(display_frame, bbox["id"], (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            
            # Aktualizuj ramkę do wyświetlenia
            self.frozen_frame = display_frame.copy()
            self.frozen_bboxes = self.bboxes.copy()
            
            # Wyświetl zaktualizowany obraz
            self.show_frame(display_frame, "Analiza komponentów na wyciętej płytce")
            print(f"Wykryto {len(self.bboxes)} obiektów")
            
            # Zapisz wyniki detekcji wraz z obrazem
            self.detection_result = {
                "image": self.preprocessed_frame.copy(),
                "detections": self.bboxes.copy()
            }

    def toggle_recording(self):
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self, "Uwaga", "Kamera nie jest uruchomiona!")
            return

        if self.recording:
            self.recording = False
            self.record_button.setText("Nagrywaj")
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                print("Nagrywanie zakończone.")
        else:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Zapisz nagranie", "", "MP4 Files (*.mp4);;AVI Files (*.avi);;All Files (*)", options=options
            )

            if not file_name:
                print("Nie wybrano pliku.")
                return

            # Sprawdzenie, czy dodano rozszerzenie
            if not (file_name.lower().endswith(".mp4") or file_name.lower().endswith(".avi")):
                file_name += ".mp4"

            # Dobór kodeka:
            if file_name.lower().endswith(".mp4"):
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Najbezpieczniejszy dla MP4
            else:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')

            # Rozdzielczość - dopasowanie do labela cap:
            resolution = (self.cap_label.width(), self.cap_label.height())

            # Próba utworzenia VideoWriter:
            self.video_writer = cv2.VideoWriter(file_name, fourcc, 30.0, resolution)

            if not self.video_writer.isOpened():
                QMessageBox.critical(self, "Błąd", f"Nie udało się otworzyć pliku do zapisu: {file_name}")
                print(f"Nie udało się otworzyć pliku: {file_name}")
                return

            self.recording = True
            self.record_button.setText("Zatrzymaj Nagrywanie")
            print(f"Nagrywanie rozpoczęte: {file_name}")

    def detect_components(self, frame):
        """Funkcja wykrywająca komponenty na obrazie"""
        # W tym miejscu frame powinien być już po preprocessingu
        
        # Konwersja BGR do RGB - dokładnie jak w test.py
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Konwersja do tensora w sposób identyczny jak w test.py
        tensor_frame = torch.tensor(rgb_frame).permute(2, 0, 1).float() / 255.0
        tensor_frame = tensor_frame.unsqueeze(0).to(self.device)
        
        print(f"Tensor shape: {tensor_frame.shape}, device: {tensor_frame.device}, model device: {next(self.model.parameters()).device}")

        with torch.no_grad():
            try:
                predictions = self.model(tensor_frame)[0]
                print("Predykcja zakończona pomyślnie")
                print(f"Predykcja zwróciła {len(predictions['boxes'])} potencjalnych obiektów")
                
                # Wypisz kilka pierwszych predykcji dla debugowania
                if len(predictions['boxes']) > 0:
                    for i in range(min(3, len(predictions['boxes']))):
                        print(f"Box {i}: {predictions['boxes'][i].tolist()}, Score: {predictions['scores'][i].item():.4f}")
            except Exception as e:
                print(f"Błąd podczas predykcji: {e}")
                import traceback
                traceback.print_exc()
                return frame, []

        detections = []
        count = 0
        
        # Ustawiam niższy próg pewności - tak jak w test.py
        confidence_threshold = 0.75  # Zmniejszono z 0.75 na 0.5
        print(f"Liczba wykrytych obiektów przed filtrowaniem: {len(predictions['boxes'])}")

        for box, score, label in zip(predictions["boxes"], predictions["scores"], predictions["labels"]):
            if score > confidence_threshold:
                count += 1
                # Konwersja koordynatów na inty
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                component_name = self.component.currentText()
                detections.append({
                    "id": f"{component_name}:{count}|Score: {score.item():.2f}", 
                    "bbox": (x1, y1, x2, y2), 
                    "score": float(score.item())
                })

        self.count_elements.setText(f"{count}")
        print(f"Liczba wykrytych obiektów po filtrowaniu (próg={confidence_threshold}): {count}")
        
        # Dodaję boxów do oryginalnego obrazu dla lepszej wizualizacji 
        # Zwiększono grubość linii z 2 na 4
        result_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            
        return result_frame, detections

    def run_preprocessing(self):
        """Ręczne uruchomienie preprocessingu obrazu"""
        if not self.frozen or self.original_frame is None:
            QMessageBox.warning(self, "Uwaga", "Najpierw załaduj zdjęcie!")
            return
            
        try:
            # Najpierw wytnij płytkę z obrazu, a następnie wykonaj preprocessing
            print("Uruchamiam preprocessing - krok 1: wykrywanie i wycinanie płytki PCB...")
            
            # Wykonaj preprocessing
            processed_img = self.preprocess_image(self.original_frame.copy())
            
            # Zapisz przetworzony obraz
            self.preprocessed_frame = processed_img.copy()
            self.is_preprocessed = True
            
            # Wyświetl przetworzony obraz
            self.show_frame(processed_img, "Płytka PCB po preprocessingu")
            
            # Aktywuj przyciski
            self.analyze_button.setEnabled(True)
            self.show_preprocessing_btn.setEnabled(True)
            
            # Ustaw tryb wyświetlania na preprocessed
            self.preprocessing_visible = True
            self.show_preprocessing_btn.setText("Pokaż detekcję")
            
            QMessageBox.information(self, "Informacja", "Płytka PCB została wycięta i przetworzona. Możesz teraz uruchomić analizę.")
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Błąd podczas preprocessingu: {str(e)}")
            import traceback
            traceback.print_exc()

    def rotate_pcb_to_horizontal(self, image, corners=None):
        """
        Obraca płytkę PCB do orientacji poziomej.
        Ulepszony algorytm uwzględniający również obrót o 180 stopni.
        
        Args:
            image: Obraz płytki PCB
            corners: Narożniki prostokąta płytki (jeśli None, zostaną wykryte)
            
        Returns:
            Obraz obrócony tak, aby płytka była w orientacji poziomej
        """
        if corners is None:
            # Jeśli nie podano narożników, spróbuj wykryć kształt płytki
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print("Nie można wykryć konturów płytki do określenia orientacji")
                return image
                
            # Znajdź największy kontur (płytka)
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            corners = cv2.boxPoints(rect)
            corners = np.int0(corners)
        
        # Jeśli mamy narożniki, użyj ich do określenia orientacji
        if corners is not None and len(corners) == 4:
            # Oblicz wymiary prostokąta
            rect = cv2.minAreaRect(corners)
            width, height = rect[1]
            angle = rect[2]
            center = rect[0]
            
            # Próbujemy wykryć, czy tekst/komponenty są w prawidłowej orientacji
            # Rozszerzone wykrywanie orientacji
            rotation_needed = False
            rotation_angle = 0
            
            # Wyświetl informacje diagnostyczne
            print(f"Wykryte wymiary płytki: {width:.1f}x{height:.1f}, kąt: {angle:.1f} stopni")
            
            # Określ czy płytka jest pionowa czy pozioma i oblicz kąt obrotu
            is_vertical = height > width
            
            # Wyświetl obrazek przed obrotem dla porównania
            display_img = image.copy()
            cv2.drawContours(display_img, [corners], 0, (0, 255, 0), 2)
            cv2.circle(display_img, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)  # Zaznacz środek
            # Zaznacz narożniki numerami dla lepszej orientacji
            for i, corner in enumerate(corners):
                cv2.putText(display_img, str(i), (int(corner[0]), int(corner[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            self.show_frame(display_img, "Płytka przed obrotem (z zaznaczonymi narożnikami)")
            
            if is_vertical:
                # Jeśli płytka jest pionowa, obracamy ją o 90 stopni
                print(f"Wykryto pionową orientację płytki (wymiary: {width:.1f}x{height:.1f})")
                rotation_needed = True
                
                # Kąt zależy od orientacji - musimy określić, w którą stronę obrócić
                if angle < -45:
                    rotation_angle = 90 + angle
                else:
                    rotation_angle = -90 + angle
                    
                print(f"Obracam płytkę o {rotation_angle:.1f} stopni do pozycji poziomej")
            else:
                # Płytka jest już pozioma, ale sprawdzamy czy nie jest odwrócona do góry nogami
                print(f"Płytka jest w orientacji poziomej (wymiary: {width:.1f}x{height:.1f})")
                
                # Sprawdzamy czy płytka nie wymaga obrotu o 180 stopni
                # To jest bardziej złożone i może wymagać analizy obrazu lub wskazówek od użytkownika
                
                # Sprawdź, czy kąt wskazuje na obrót względem osi poziomej
                if abs(angle) > 45:
                    rotation_needed = True
                    rotation_angle = 180  # Obrót o 180 stopni
                    print(f"Płytka prawdopodobnie jest odwrócona, obracam o 180 stopni")
                else:
                    # Nawet jeśli kąt nie wskazuje na obrót, dajmy użytkownikowi możliwość wyboru
                    result = QMessageBox.question(self, "Orientacja płytki", 
                                               "Czy płytka jest w prawidłowej orientacji (tekst/komponenty nie są odwrócone)?",
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                    
                    if result == QMessageBox.No:
                        rotation_needed = True
                        rotation_angle = 180
                        print("Użytkownik wskazał, że płytka jest odwrócona. Obracam o 180 stopni.")
            
            # Wykonaj obrót, jeśli jest potrzebny
            if rotation_needed:
                # Pobierz wymiary obrazu
                h, w = image.shape[:2]
                
                # Oblicz macierz obrotu
                M = cv2.getRotationMatrix2D((w/2, h/2), rotation_angle, 1.0)
                
                # Oblicz nowe wymiary po obrocie
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                
                # Nowe wymiary obrazu po obrocie
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))
                
                # Dopasuj macierz translacji do nowych wymiarów
                M[0, 2] += (new_w / 2) - (w / 2)
                M[1, 2] += (new_h / 2) - (h / 2)
                
                # Wykonaj obrót
                rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, 
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
                
                # Jeśli obraz został obrócony z pionowego na poziomy, przytnij ewentualne czarne/białe marginesy
                if is_vertical:
                    # Konwertuj na skalę szarości i znajdź niepuste obszary
                    gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
                    _, thresh_rot = cv2.threshold(gray_rotated, 240, 255, cv2.THRESH_BINARY_INV)
                    contours_rot, _ = cv2.findContours(thresh_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours_rot:
                        largest_contour_rot = max(contours_rot, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour_rot)
                        # Dodaj małe marginesy
                        margin = 10
                        x = max(0, x - margin)
                        y = max(0, y - margin)
                        w = min(rotated.shape[1] - x, w + 2*margin)
                        h = min(rotated.shape[0] - y, h + 2*margin)
                        rotated = rotated[y:y+h, x:x+w]
                
                # Pokaż obrócony obraz
                self.show_frame(rotated, f"Płytka PCB po obróceniu o {rotation_angle:.1f} stopni")
                
                return rotated
                
        # Jeśli nie można określić orientacji lub obrót nie jest potrzebny, zwróć oryginalny obraz
        return image

    def detect_pcb_contour(self, image):
        """
        Wykrywa kontury płytki PCB na obrazie i zwraca wyciętą płytkę.
        Optymalizacja dla zielonej płytki na białym tle.
        
        Algorytm:
        1. Konwersja do przestrzeni HSV i segmentacja koloru zielonego
        2. Operacje morfologiczne w celu poprawy konturu płytki
        3. Wykrycie największego konturu (płytki PCB)
        4. Wycięcie płytki
        5. Obrócenie płytki do orientacji poziomej
        
        Zwraca:
        - Wyciętą płytkę PCB jako obraz w orientacji poziomej
        - Narożniki płytki PCB
        """
        print("Wykrywam zieloną płytkę na białym tle...")
        
        # Tworzymy kopię obrazu do pokazania wyników
        display_img = image.copy()
        
        # 1. Konwersja do przestrzeni HSV i segmentacja koloru zielonego
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Zakres koloru zielonego dla płytki PCB - szerszy zakres dla większej tolerancji
        # Można dostosować te wartości w zależności od konkretnego odcienia zieleni płytki
        lower_green = np.array([35, 25, 25])   # Niższe nasycenie i jasność dla wykrycia również ciemniejszych odcieni
        upper_green = np.array([90, 255, 255]) # Szerszy zakres odcieni zieleni
        
        # Utwórz maskę koloru zielonego
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Pokaż maskę dla diagnostyki
        self.show_frame(cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR), "Maska koloru zielonego")
        
        # 2. Operacje morfologiczne w celu poprawy konturu płytki
        # Zastosuj dużą operację zamknięcia, aby wypełnić wszystkie dziury w masce
        kernel_close = np.ones((25, 25), np.uint8)
        mask_closed = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Zastosuj mniejszą operację otwarcia, aby usunąć mały szum
        kernel_open = np.ones((5, 5), np.uint8)
        mask_processed = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)
        
        # Pokaż przetworzoną maskę
        self.show_frame(cv2.cvtColor(mask_processed, cv2.COLOR_GRAY2BGR), "Maska po operacjach morfologicznych")
        
        # 3. Wykrycie największego konturu (płytki PCB)
        contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Jeśli nie znaleziono żadnych konturów, spróbuj alternatywną metodę
        if not contours:
            print("Nie znaleziono żadnych konturów, próbuję alternatywną metodę...")
            return self.detect_pcb_alternative_robust(image)
        
        # Wybierz największy kontur (prawdopodobnie płytka PCB)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Jeśli kontur jest za mały, spróbuj alternatywną metodę
        if cv2.contourArea(largest_contour) < (image.shape[0] * image.shape[1] * 0.05):  # Minimalnie 5% obszaru obrazu
            print(f"Znaleziony kontur jest za mały ({cv2.contourArea(largest_contour)} pikseli), próbuję alternatywną metodę...")
            return self.detect_pcb_alternative_robust(image)
        
        # Narysuj kontur na obrazie dla wizualizacji
        cv2.drawContours(display_img, [largest_contour], -1, (0, 255, 0), 3)
        self.show_frame(display_img, "Wykryty kontur płytki PCB")
        
        # Znajdź prostokąt ograniczający kontur
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Narysuj prostokąt na obrazie dla wizualizacji
        cv2.drawContours(display_img, [box], 0, (0, 0, 255), 3)
        self.show_frame(display_img, "Ograniczający prostokąt płytki PCB")
        
        # Zapisz narożniki
        self.pcb_corners = box
        
        # 4. Wycięcie płytki z obrazu
        # Oblicz minimalne i maksymalne współrzędne prostokąta
        h, w = image.shape[:2]
        margin = 5  # Dodatkowy margines wokół płytki
        
        # Znajdź minimalne i maksymalne współrzędne
        min_x = max(0, int(np.min(box[:, 0])) - margin)
        min_y = max(0, int(np.min(box[:, 1])) - margin)
        max_x = min(w - 1, int(np.max(box[:, 0])) + margin)
        max_y = min(h - 1, int(np.max(box[:, 1])) + margin)
        
        # Wytnij płytkę z obrazu
        cropped_pcb = image[min_y:max_y, min_x:max_x]
        
        print(f"Wycięto płytkę PCB o wymiarach: {cropped_pcb.shape}")
        
        # 5. Obróć płytkę do orientacji poziomej
        # Przeliczymy narożniki względem wyciętego obrazu
        adjusted_corners = box.copy()
        adjusted_corners[:, 0] -= min_x
        adjusted_corners[:, 1] -= min_y
        
        # Obróć wyciętą płytkę
        rotated_pcb = self.rotate_pcb_to_horizontal(cropped_pcb, adjusted_corners)
        
        # Pokaż wycięty i obrócony obraz
        self.show_frame(rotated_pcb, "Wycięta i obrócona płytka PCB")
        
        # Daj użytkownikowi szansę potwierdzenia wykrytej płytki
        result = QMessageBox.question(self, "Wykrywanie płytki PCB", 
                                     "Czy wykryta płytka PCB jest poprawna?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        
        if result == QMessageBox.Yes:
            return rotated_pcb, self.pcb_corners
        else:
            print("Użytkownik odrzucił wykrytą płytkę - używam alternatywnej metody")
            return self.detect_pcb_alternative_robust(image)
            
    def detect_pcb_alternative_robust(self, image):
        """
        Ulepszona alternatywna metoda wykrywania płytki PCB, bardziej odporna
        i specjalnie zoptymalizowana dla zielonej płytki na białym tle.
        
        Zwraca:
        - Wyciętą płytkę PCB jako obraz w orientacji poziomej
        - Narożniki płytki PCB
        """
        print("Używam ulepszonych metod wykrywania płytki na białym tle...")
        
        # 1. Konwersja do skali szarości
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Pokaż obraz w skali szarości
        self.show_frame(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "Obraz w skali szarości")
        
        # 2. Progowanie adaptacyjne - dobre dla różnych warunków oświetleniowych
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 51, 10)
        
        # Pokaż obraz po progowaniu
        self.show_frame(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "Obraz po progowaniu adaptacyjnym")
        
        # 3. Zastosuj operacje morfologiczne, aby usunąć szum i wzmocnić kontury
        kernel = np.ones((15, 15), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        
        # Pokaż obraz po operacjach morfologicznych
        self.show_frame(cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR), "Obraz po operacjach morfologicznych")
        
        # 4. Znajdź kontury
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Jeśli nie znaleziono żadnych konturów, użyj innego podejścia z różnicą kolorów
        if not contours:
            return self.detect_pcb_color_difference(image)
            
        # Posortuj kontury według obszaru (od największego do najmniejszego)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Sprawdź każdy kontur pod kątem prostokątnego kształtu typowego dla płytki PCB
        for contour in contours[:3]:  # Sprawdź tylko 3 największe kontury
            # Pomiń zbyt małe kontury
            if cv2.contourArea(contour) < (image.shape[0] * image.shape[1] * 0.05):
                continue
                
            # Znajdź ograniczający prostokąt
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Sprawdź proporcje prostokąta
            width = np.linalg.norm(box[0] - box[1])
            height = np.linalg.norm(box[1] - box[2])
            
            if width == 0 or height == 0:
                continue
                
            aspect_ratio = max(width, height) / min(width, height)
            
            # Płytki PCB zazwyczaj mają proporcje między 1:1 a 4:1
            if 1.0 <= aspect_ratio <= 4.0:
                # Dla wizualizacji - rysujemy prostokąt na obrazie
                display_img = image.copy()
                cv2.drawContours(display_img, [box], 0, (0, 0, 255), 3)
                self.show_frame(display_img, "Wykryty kontur płytki PCB (metoda alternatywna)")
                
                # Zapisz narożniki
                self.pcb_corners = box
                
                # Oblicz granice prostokąta z dodatkowym marginesem
                h, w = image.shape[:2]
                margin = 5
                
                # Znajdź minimalne i maksymalne współrzędne
                min_x = max(0, int(np.min(box[:, 0])) - margin)
                min_y = max(0, int(np.min(box[:, 1])) - margin)
                max_x = min(w - 1, int(np.max(box[:, 0])) + margin)
                max_y = min(h - 1, int(np.max(box[:, 1])) + margin)
                
                # Wytnij płytkę z obrazu
                cropped_pcb = image[min_y:max_y, min_x:max_x]
                
                print(f"Wycięto płytkę PCB o wymiarach: {cropped_pcb.shape}, proporcje: {aspect_ratio:.2f}")
                
                # Przeliczymy narożniki względem wyciętego obrazu
                adjusted_corners = box.copy()
                adjusted_corners[:, 0] -= min_x
                adjusted_corners[:, 1] -= min_y
                
                # Obróć wyciętą płytkę
                rotated_pcb = self.rotate_pcb_to_horizontal(cropped_pcb, adjusted_corners)
                
                # Pokaż wycięty i obrócony obraz przed kontynuacją
                self.show_frame(rotated_pcb, "Wycięta i obrócona płytka PCB (metoda alternatywna)")
                
                # Daj użytkownikowi szansę potwierdzenia wykrytych konturów
                result = QMessageBox.question(self, "Wykrywanie płytki PCB", 
                                             "Czy wykryta płytka PCB jest poprawna?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                
                if result == QMessageBox.Yes:
                    return rotated_pcb, box
        
        # Jeśli nie znaleziono odpowiedniego konturu, spróbuj inną metodę
        return self.detect_pcb_color_difference(image)
        
    def detect_pcb_color_difference(self, image):
        """
        Metoda wykrywania płytki PCB bazująca bezpośrednio na różnicy kolorów
        między białym tłem a zieloną płytką.
        
        Zwraca:
        - Wyciętą płytkę PCB jako obraz w orientacji poziomej
        - Narożniki płytki PCB
        """
        print("Używam metody różnicy kolorów do wykrycia płytki PCB...")
        
        # Rozdziel obraz na kanały kolorów
        b, g, r = cv2.split(image)
        
        # Dla zielonej płytki na białym tle, kanał zielony będzie miał wyższe wartości 
        # na płytce niż kanały czerwony i niebieski
        
        # Oblicz różnicę między kanałem zielonym a średnią pozostałych kanałów
        green_diff = cv2.subtract(g, cv2.addWeighted(b, 0.5, r, 0.5, 0))
        
        # Progowanie różnicy
        _, mask = cv2.threshold(green_diff, 20, 255, cv2.THRESH_BINARY)
        
        # Pokaż maskę różnicy kolorów
        self.show_frame(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), "Maska różnicy kolorów")
        
        # Operacje morfologiczne, aby poprawić maskę
        kernel = np.ones((20, 20), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        
        # Pokaż przetworzoną maskę
        self.show_frame(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), "Przetworzona maska różnicy kolorów")
        
        # Znajdź kontury
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Wybierz największy kontur
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Jeśli kontur jest za mały, przejdź do ręcznego określania
            if cv2.contourArea(largest_contour) < (image.shape[0] * image.shape[1] * 0.05):
                return self.manual_pcb_detection(image)
                
            # Znajdź ograniczający prostokąt
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Narysuj prostokąt na obrazie
            display_img = image.copy()
            cv2.drawContours(display_img, [box], 0, (0, 0, 255), 3)
            self.show_frame(display_img, "Wykryty kontur płytki PCB (metoda różnicy kolorów)")
            
            # Zapisz narożniki
            self.pcb_corners = box
            
            # Oblicz granice prostokąta
            h, w = image.shape[:2]
            margin = 5
            
            # Znajdź minimalne i maksymalne współrzędne
            min_x = max(0, int(np.min(box[:, 0])) - margin)
            min_y = max(0, int(np.min(box[:, 1])) - margin)
            max_x = min(w - 1, int(np.max(box[:, 0])) + margin)
            max_y = min(h - 1, int(np.max(box[:, 1])) + margin)
            
            # Wytnij płytkę z obrazu
            cropped_pcb = image[min_y:max_y, min_x:max_x]
            
            print(f"Wycięto płytkę PCB o wymiarach: {cropped_pcb.shape}")
            
            # Przeliczymy narożniki względem wyciętego obrazu
            adjusted_corners = box.copy()
            adjusted_corners[:, 0] -= min_x
            adjusted_corners[:, 1] -= min_y
            
            # Obróć wyciętą płytkę
            rotated_pcb = self.rotate_pcb_to_horizontal(cropped_pcb, adjusted_corners)
            
            # Pokaż wycięty i obrócony obraz przed kontynuacją
            self.show_frame(rotated_pcb, "Wycięta i obrócona płytka PCB (metoda różnicy kolorów)")
            
            # Daj użytkownikowi szansę potwierdzenia wykrytych konturów
            result = QMessageBox.question(self, "Wykrywanie płytki PCB", 
                                         "Czy wykryta płytka PCB jest poprawna?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            
            if result == QMessageBox.Yes:
                return rotated_pcb, box
        
        # Jeśli nie znaleziono odpowiedniego konturu, przejdź do ręcznego określania
        return self.manual_pcb_detection(image)
        
    def manual_pcb_detection(self, image):
        """
        Ostatnia metoda, gdy automatyczne wykrywanie zawiedzie.
        Pozwala użytkownikowi zdecydować, czy użyć całego obrazu.
        
        W przyszłości można tu zaimplementować ręczne zaznaczanie obszaru płytki.
        
        Zwraca:
        - Obraz (oryginalny lub przycięty) w orientacji poziomej
        - Narożniki (lub None)
        """
        print("Wszystkie metody automatyczne zawiodły. Pytam użytkownika o decyzję.")
        
        # Daj użytkownikowi wybór
        result = QMessageBox.question(self, "Wykrywanie płytki PCB", 
                                     "Nie udało się automatycznie wykryć płytki PCB.\n"
                                     "Czy chcesz użyć całego obrazu?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        
        if result == QMessageBox.Yes:
            print("Użytkownik wybrał użycie całego obrazu")
            
            # Nawet jeśli używamy całego obrazu, spróbujmy go obrócić do pozycji poziomej
            # Używamy none jako corners, co spowoduje próbę automatycznego wykrycia
            rotated_image = self.rotate_pcb_to_horizontal(image)
            return rotated_image, None
        else:
            # Tutaj można by dodać interfejs do ręcznego zaznaczania obszaru
            # Na razie po prostu używamy domyślnego przycinania o 10% z każdej strony
            print("Używam domyślnego przycinania obrazu")
            
            h, w = image.shape[:2]
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            
            cropped = image[margin_y:h-margin_y, margin_x:w-margin_x]
            
            if cropped.size == 0:  # Sprawdź, czy przycinanie nie dało pustego obrazu
                print("Przycinanie dało pusty obraz - używam oryginału")
                rotated_image = self.rotate_pcb_to_horizontal(image)
                return rotated_image, None
                
            # Pokaż przycięty obraz
            self.show_frame(cropped, "Domyślnie przycięty obraz")
            
            # Spróbuj obrócić przycięty obraz do pozycji poziomej
            rotated_cropped = self.rotate_pcb_to_horizontal(cropped)
            self.show_frame(rotated_cropped, "Przycięty i obrócony obraz")
            
            # Pytanie, czy jest ok
            confirm = QMessageBox.question(self, "Domyślne przycinanie", 
                                          "Czy domyślnie przycięty obraz jest odpowiedni?",
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                                          
            if confirm == QMessageBox.Yes:
                return rotated_cropped, None
            else:
                print("Użytkownik odrzucił domyślne przycinanie - używam oryginalnego obrazu")
                rotated_image = self.rotate_pcb_to_horizontal(image)
                return rotated_image, None

    def preprocess_image(self, image):
        """
        Rozszerzony preprocessing obrazu dla systemów AOI.
        0) Wykrycie i wycięcie płytki PCB
        1) Korekcja optyczna dystorsji
        2) Globalna normalizacja histogramu
        3) Usuwanie szumu (medianowy lub Gauss) i filtr adaptacyjny bilateralny
        4) Adaptacyjne wyrównanie kontrastu (CLAHE) na kanale LAB
        5) Regulacja kontrastu i jasności
        6) Rejestracja obrazu za pomocą fiduciali
        7) Wstępna segmentacja (HSV i progi)

        Zwraca przetworzony obraz BGR, a maski segmentacji są w self.masks.
        """
        # 0. Wykrycie i wycięcie płytki PCB
        print("Wykrywanie i wycinanie płytki PCB...")
        pcb_image, corners = self.detect_pcb_contour(image)
        
        # Jeśli nie znaleziono płytki lub użytkownik odrzucił detekcję, użyj oryginalnego obrazu
        if pcb_image is None or corners is None:
            print("Nie znaleziono poprawnej płytki PCB - kontynuuję z oryginalnym obrazem")
            pcb_image = image.copy()
        
        # Zapisz oryginalne wymiary obrazu z płytką do późniejszego odniesienia
        self.pcb_image_original = pcb_image.copy()
        
        # 1. Korekcja optyczna (dystorsja)
        if hasattr(self, 'camera_matrix') and hasattr(self, 'dist_coeffs'):
            pcb_image = cv2.undistort(pcb_image, self.camera_matrix, self.dist_coeffs)
            
        # 2. Globalna normalizacja histogramu
        normalized = cv2.normalize(pcb_image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Pokaż obraz po normalizacji
        self.show_frame(normalized, "Obraz po normalizacji histogramu")

        # 3. Odszumianie
        # 3a. Filtr medianowy lub Gaussowski
        if getattr(self, 'use_gaussian', False):
            denoised = cv2.GaussianBlur(normalized, (3,3), 0)
        else:
            denoised = cv2.medianBlur(normalized, 3)
        # 3b. Adaptacyjny filtr bilateralny
        denoised = cv2.bilateralFilter(denoised, d=5, sigmaColor=75, sigmaSpace=75)
        
        # Pokaż obraz po odszumianiu
        self.show_frame(denoised, "Obraz po odszumianiu")

        # 4. CLAHE na kanale L w przestrzeni LAB
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Pokaż obraz po CLAHE
        self.show_frame(processed, "Obraz po CLAHE")

        # 5. Regulacja kontrastu i jasności
        result = cv2.convertScaleAbs(processed, alpha=1.1, beta=5)
        
        # Pokaż obraz po regulacji kontrastu
        self.show_frame(result, "Obraz po regulacji kontrastu")

        # 6. Rejestracja obrazu (alignment) - fiduciale
        if hasattr(self, 'fiducial_detector'):
            pts_src = self.fiducial_detector.detect(pcb_image)
            pts_dst = self.fiducial_detector.reference_points
            if len(pts_src) >= 3:
                M, _ = cv2.estimateAffinePartial2D(np.array(pts_src), np.array(pts_dst))
                result = cv2.warpAffine(result, M, (result.shape[1], result.shape[0]))

        # 7. Wstępna segmentacja
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        # maska soldermask
        lower_sm, upper_sm = getattr(self, 'soldermask_lower', (35, 50, 50)), getattr(self, 'soldermask_upper', (85, 255, 255))
        sm_mask = cv2.inRange(hsv, lower_sm, upper_sm)
        # maska padów (biały lub metaliczny)
        pad_mask = cv2.inRange(result, getattr(self, 'pad_lower', (200,200,200)), getattr(self, 'pad_upper', (255,255,255)))
        # maska otworów
        hole_mask = cv2.bitwise_not(sm_mask)
        
        # Pokaż maski dla diagnostyki
        self.show_frame(cv2.cvtColor(sm_mask, cv2.COLOR_GRAY2BGR), "Maska soldermask")
        self.show_frame(cv2.cvtColor(pad_mask, cv2.COLOR_GRAY2BGR), "Maska padów")
        self.show_frame(cv2.cvtColor(hole_mask, cv2.COLOR_GRAY2BGR), "Maska otworów")

        # zapisz maski
        self.masks = {'soldermask': sm_mask, 'pad': pad_mask, 'hole': hole_mask}

        # zachowanie wyniku
        self.preprocessed_frame = result
        return result

    def preprocess_image_skip_detection(self, image):
        """
        Wykonuje preprocessing obrazu PCB bez etapu wykrywania płytki.
        Ten obraz powinien być już przyciętą płytką PCB.
        
        1) Korekcja optyczna dystorsji
        2) Globalna normalizacja histogramu
        3) Usuwanie szumu (medianowy lub Gauss) i filtr adaptacyjny bilateralny
        4) Adaptacyjne wyrównanie kontrastu (CLAHE) na kanale LAB
        5) Regulacja kontrastu i jasności
        6) Rejestracja obrazu za pomocą fiduciali
        7) Wstępna segmentacja (HSV i progi)

        Zwraca przetworzony obraz BGR, a maski segmentacji są w self.masks.
        """
        # Zapisz oryginalne wymiary obrazu z płytką do późniejszego odniesienia
        self.pcb_image_original = image.copy()
        
        # 1. Korekcja optyczna (dystorsja)
        if hasattr(self, 'camera_matrix') and hasattr(self, 'dist_coeffs'):
            image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
            
        # 2. Globalna normalizacja histogramu
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Pokaż obraz po normalizacji
        self.show_frame(normalized, "Obraz po normalizacji histogramu")

        # 3. Odszumianie
        # 3a. Filtr medianowy lub Gaussowski
        if getattr(self, 'use_gaussian', False):
            denoised = cv2.GaussianBlur(normalized, (3,3), 0)
        else:
            denoised = cv2.medianBlur(normalized, 3)
        # 3b. Adaptacyjny filtr bilateralny
        denoised = cv2.bilateralFilter(denoised, d=5, sigmaColor=75, sigmaSpace=75)
        
        # Pokaż obraz po odszumianiu
        self.show_frame(denoised, "Obraz po odszumianiu")

        # 4. CLAHE na kanale L w przestrzeni LAB
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Pokaż obraz po CLAHE
        self.show_frame(processed, "Obraz po CLAHE")

        # 5. Regulacja kontrastu i jasności
        result = cv2.convertScaleAbs(processed, alpha=1.1, beta=5)
        
        # Pokaż obraz po regulacji kontrastu
        self.show_frame(result, "Obraz po regulacji kontrastu")

        # 6. Rejestracja obrazu (alignment) - fiduciale
        if hasattr(self, 'fiducial_detector'):
            pts_src = self.fiducial_detector.detect(image)
            pts_dst = self.fiducial_detector.reference_points
            if len(pts_src) >= 3:
                M, _ = cv2.estimateAffinePartial2D(np.array(pts_src), np.array(pts_dst))
                result = cv2.warpAffine(result, M, (result.shape[1], result.shape[0]))

        # 7. Wstępna segmentacja
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        # maska soldermask
        lower_sm, upper_sm = getattr(self, 'soldermask_lower', (35, 50, 50)), getattr(self, 'soldermask_upper', (85, 255, 255))
        sm_mask = cv2.inRange(hsv, lower_sm, upper_sm)
        # maska padów (biały lub metaliczny)
        pad_mask = cv2.inRange(result, getattr(self, 'pad_lower', (200,200,200)), getattr(self, 'pad_upper', (255,255,255)))
        # maska otworów
        hole_mask = cv2.bitwise_not(sm_mask)
        
        # Pokaż maski dla diagnostyki
        self.show_frame(cv2.cvtColor(sm_mask, cv2.COLOR_GRAY2BGR), "Maska soldermask")
        self.show_frame(cv2.cvtColor(pad_mask, cv2.COLOR_GRAY2BGR), "Maska padów")
        self.show_frame(cv2.cvtColor(hole_mask, cv2.COLOR_GRAY2BGR), "Maska otworów")

        # zapisz maski
        self.masks = {'soldermask': sm_mask, 'pad': pad_mask, 'hole': hole_mask}

        # zachowanie wyniku
        self.preprocessed_frame = result
        return result

    def update_frame(self):
        # Jeżeli analiza dotyczy statycznego obrazu, to nie próbujemy pobierać klatek z kamery
        if self.cap is None and self.frozen and self.frozen_frame is not None:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.critical(self, "Błąd", "Nie udało się pobrać klatki z kamery.")
            self.stop_camera()
            return

        self.frame_count += 1
        original_frame = frame.copy()

        if self.analyze:
            # Najpierw wykrywamy i wycinamy płytkę PCB - dostosowane dla zielonego PCB na białym tle
            print("Wykrywanie płytki PCB z obrazu kamery...")
            pcb_frame, corners = self.detect_pcb_contour(original_frame)
            
            # Zachowaj referencję do oryginalnego obrazu przed preprocessingiem
            if corners is not None:
                # Użytkownik zaakceptował wykrytą płytkę
                self.original_frame = pcb_frame.copy()
                
                # Narysuj kontur płytki na oryginalnym obrazie dla wizualizacji
                display_img = original_frame.copy()
                cv2.drawContours(display_img, [corners], 0, (0, 255, 0), 3)
                self.show_frame(display_img, "Wykryta płytka PCB")
            else:
                # Użytkownik odrzucił wykrytą płytkę lub nie została znaleziona
                self.original_frame = original_frame.copy()
                
            # Wykonaj pełny preprocessing na wyciętej płytce
            preprocessed_frame = self.preprocess_image_skip_detection(self.original_frame.copy())
            
            # Następnie wykonujemy detekcję na przetworzonym obrazie
            processed_with_detections, detections = self.detect_components(preprocessed_frame)  # Pobieramy detekcje
            self.update_component_list(detections)  # Aktualizujemy listę ID komponentów

            # Rysowanie bounding boxów na przetworzonym obrazie
            display_frame = preprocessed_frame.copy()  # Używamy przetworzonego obrazu płytki
            for bbox in self.bboxes:
                x1, y1, x2, y2 = bbox["bbox"]
                color = bbox["color"]
                # Zwiększono grubość ramki dla lepszej widoczności
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 4)
                # Dodaj etykietę z ID
                cv2.putText(display_frame, bbox["id"], (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            # Zamrażamy klatkę po analizie - używamy przetworzonego obrazu z boxami
            self.frozen_frame = display_frame.copy()
            self.frozen_bboxes = self.bboxes.copy()
            self.frozen = True
            
            # Zachowujemy przetworzony obraz do podglądu
            self.preprocessed_frame = preprocessed_frame.copy()
            self.is_preprocessed = True
            
            # Zapisz wyniki detekcji wraz z obrazem
            self.detection_result = {
                "image": preprocessed_frame.copy(),
                "detections": self.bboxes.copy()
            }
            
            # Pokaż wynik detekcji
            self.show_frame(display_frame, f"Wykryto {len(self.bboxes)} komponentów")

        # Jeśli analiza została zatrzymana, pokazujemy zamrożoną klatkę
        elif self.frozen:
            frame = self.frozen_frame.copy()
        else:
            # Jeśli nie ma analizy ani zamrożonej klatki, pokaż bieżącą klatkę
            frame = original_frame
            
            # Dodaj informację, że analiza jest wyłączona
            cv2.putText(frame, "Analiza wyłączona - kliknij 'Analiza' aby rozpocząć",
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Wyświetlanie obrazu
        frame_resized = cv2.resize(frame, (self.cap_label.width(), self.cap_label.height()), interpolation=cv2.INTER_LINEAR)
        h, w, ch = frame_resized.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_resized.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.cap_label.setPixmap(QPixmap.fromImage(qimg))

        if self.recording and self.video_writer:
            self.video_writer.write(frame)  # Zapisujemy wideo

    def show_frame(self, frame, info_text=None):
        """Funkcja pomocnicza do wyświetlania obrazu - poprawiona skalowanie"""
        if frame is None:
            print("BŁĄD: Próba wyświetlenia pustej ramki!")
            return
            
        print(f"Wyświetlam ramkę o wymiarach: {frame.shape}")
        
        # Dodanie informacji tekstowej na obrazie jeśli podano
        display_frame = frame.copy()
        if info_text:
            cv2.putText(
                display_frame, 
                info_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (0, 0, 255), 
                2
            )
        
        # Używamy resize_with_aspect_ratio zamiast zwykłego resize
        # aby zachować proporcje obrazu
        frame_resized = self.resize_with_aspect_ratio(
            display_frame, 
            self.cap_label.width(), 
            self.cap_label.height()
        )
        
        h, w, ch = frame_resized.shape
        bytes_per_line = ch * w
        
        # Zapewniamy poprawny format koloru (BGR dla OpenCV)
        qimg = QImage(frame_resized.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.cap_label.setPixmap(QPixmap.fromImage(qimg))

        if self.recording and self.video_writer:
            self.video_writer.write(frame)
            
    def toggle_preprocessing_view(self):
        """Przełącza widok między oryginalnym obrazem a obrazem po preprocessingu"""
        if not self.frozen or self.preprocessed_frame is None:
            QMessageBox.warning(self, "Uwaga", "Najpierw załaduj zdjęcie i wykonaj preprocessing!")
            return
            
        if self.preprocessing_visible:
            # Jeśli pokazujemy preprocessed, przełącz na obraz z boxami
            if hasattr(self, 'detection_result') and self.detection_result is not None:
                # Użyj zapisanego obrazu z boxami
                display_frame = self.frozen_frame.copy()
                self.show_frame(display_frame, "Obraz z detekcją")
                self.show_preprocessing_btn.setText("Pokaż preprocessing")
            else:
                # Jeśli nie mamy zapisanego obrazu z boxami, narysuj je na preprocessed
                display_frame = self.preprocessed_frame.copy()
                for bbox in self.bboxes:
                    x1, y1, x2, y2 = bbox["bbox"]
                    color = bbox["color"]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 4)
                    cv2.putText(display_frame, bbox["id"], (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                self.show_frame(display_frame, "Obraz z detekcją")
                self.show_preprocessing_btn.setText("Pokaż preprocessing")
        else:
            # Jeśli pokazujemy obraz z boxami, przełącz na czysty preprocessed
            preprocessed_view = self.preprocessed_frame.copy()
            self.show_frame(preprocessed_view, "Obraz po preprocessingu")
            self.show_preprocessing_btn.setText("Pokaż detekcję")
            
        self.preprocessing_visible = not self.preprocessing_visible

    def show_frozen_frame(self):
        """Wyświetla ostatnią zamrożoną klatkę"""
        if self.frozen_frame is not None:
            self.show_frame(self.frozen_frame)

    def clear_image(self):
        """Resetuje obraz i wznawia działanie kamery"""
        self.frozen = False
        self.frozen_frame = None
        self.original_frame = None
        self.cap_label.clear()  # Czyści obrazek
        self.bboxes = []  # Czyści bounding boxy
        self.component_list.clear()  # Czyści listę komponentów
        self.count_elements.setText("")  # Czyści licznik elementów
        
        # Resetujemy analizę
        if self.analyze:
            self.analyze = False
            self.analyze_button.setText("Analiza")

    def update_component_list(self, detections):
        """Aktualizuje listę ID komponentów na podstawie wykrytych obiektów"""
        self.component_list.clear()  # Czyści starą listę
        self.bboxes = []  # Lista boxów

        print(f"Otrzymano {len(detections)} detekcji do aktualizacji listy")
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            id_ = f"ID:{i+1}|Score: {detection['score']:.2f}"
            self.component_list.addItem(id_)  # Dodajemy ID do listy 

            # Dodajemy bbox do listy z domyślnym kolorem
            self.bboxes.append({"id": id_, "bbox": (x1, y1, x2, y2), "color": (0, 255, 0), "score": detection["score"]})  # zielony
        
        print(f"Zaktualizowano listę komponentów, dodano {len(self.bboxes)} elementów")

    def highlight_bbox(self, item):
        """Zmienia kolor bounding boxa po kliknięciu w ID na liście"""
        clicked_id = item.text()  # Pobieramy ID
        print(f"Wybrano element: {clicked_id}")

        updated = False  # Flaga sprawdzająca, czy znaleziono ID
        for bbox in self.bboxes:
            if bbox["id"] == clicked_id:
                bbox["color"] = (255, 0, 0)  # Zmień kolor na czerwony
                updated = True
                print(f"Znaleziono i wyróżniono bbox: {bbox['bbox']}")
            else:
                bbox["color"] = (0, 255, 0)  # Kolor zielony dla pozostałych

        if updated:
            if self.cap is not None and self.cap.isOpened() and not self.frozen:
                # Jeśli mamy aktywną kamerę i nie jest zamrożona, trigger aktualizację
                self.update_frame()
            elif self.frozen and self.preprocessed_frame is not None:
                # Jeśli mamy statyczny obraz, aktualizujemy go ręcznie
                # Pobierz przetworzony obraz bez boxów
                display_frame = self.preprocessed_frame.copy()
                
                # Aktualizujemy frozen_bboxes
                self.frozen_bboxes = self.bboxes.copy()
                
                # Rysowanie bounding boxów na obrazie
                for bbox in self.frozen_bboxes:
                    x1, y1, x2, y2 = bbox["bbox"]
                    color = bbox["color"]
                    # Zwiększono grubość ramki dla lepszej widoczności
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 4)
                    # Etykieta z ID
                    cv2.putText(display_frame, bbox["id"], (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                
                # Zapisz zaktualizowaną klatkę
                self.frozen_frame = display_frame.copy()
                
                # Wyświetl zaktualizowany obraz
                self.show_frame(display_frame, "Komponent podświetlony")
                print("Zaktualizowano wyświetlanie statycznego obrazu")
                
            self.cap_label.repaint()  # Wymuś ponowne narysowanie

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()