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
            display_frame = self.original_frame.copy()  # Rysuj na oryginalnym obrazie dla lepszej czytelności
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
            self.show_frame(display_frame)
            print(f"Wykryto {len(self.bboxes)} obiektów")

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
            # Wykonaj preprocessing
            print("Uruchamiam preprocessing...")
            processed_img = self.preprocess_image(self.original_frame.copy())
            
            # Zapisz przetworzony obraz
            self.preprocessed_frame = processed_img.copy()
            self.is_preprocessed = True
            
            # Wyświetl przetworzony obraz
            self.show_frame(processed_img)
            
            # Aktywuj przyciski
            self.analyze_button.setEnabled(True)
            self.show_preprocessing_btn.setEnabled(True)
            
            QMessageBox.information(self, "Informacja", "Preprocessing zakończony. Możesz teraz uruchomić analizę.")
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Błąd podczas preprocessingu: {str(e)}")
            import traceback
            traceback.print_exc()

    def preprocess_image(self, image):
        """
        Rozszerzony preprocessing obrazu dla systemów AOI.
        1) Korekcja optyczna dystorsji
        2) Globalna normalizacja histogramu
        3) Usuwanie szumu (medianowy lub Gauss) i filtr adaptacyjny bilateralny
        4) Adaptacyjne wyrównanie kontrastu (CLAHE) na kanale LAB
        5) Regulacja kontrastu i jasności
        6) Rejestracja obrazu za pomocą fiduciali
        7) Wstępna segmentacja (HSV i progi)

        Zwraca przetworzony obraz BGR, a maski segmentacji są w self.masks.
        """
        # 1. Korekcja optyczna (dystorsja)
        if hasattr(self, 'camera_matrix') and hasattr(self, 'dist_coeffs'):
            image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
        # 2. Globalna normalizacja histogramu
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        # 3. Odszumianie
        # 3a. Filtr medianowy lub Gaussowski
        if getattr(self, 'use_gaussian', False):
            denoised = cv2.GaussianBlur(normalized, (3,3), 0)
        else:
            denoised = cv2.medianBlur(normalized, 3)
        # 3b. Adaptacyjny filtr bilateralny
        denoised = cv2.bilateralFilter(denoised, d=5, sigmaColor=75, sigmaSpace=75)

        # 4. CLAHE na kanale L w przestrzeni LAB
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 5. Regulacja kontrastu i jasności
        result = cv2.convertScaleAbs(processed, alpha=1.1, beta=5)

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
            # Najpierw wykonujemy preprocessing obrazu
            preprocessed_frame = self.preprocess_image(original_frame)
            
            # Następnie wykonujemy detekcję na przetworzonym obrazie
            frame, detections = self.detect_components(preprocessed_frame)  # Pobieramy detekcje
            self.update_component_list(detections)  # Aktualizujemy listę ID komponentów

            # **Zamrażamy klatkę po analizie**  
            self.frozen_frame = frame.copy()  
            self.frozen_bboxes = self.bboxes.copy()  
            self.frozen = True  # Ustawiamy zamrożenie
            
            # Zachowujemy przetworzony obraz do podglądu
            self.preprocessed_frame = preprocessed_frame.copy()
            self.is_preprocessed = True

        # **Jeśli analiza została zatrzymana, pokazujemy zamrożoną klatkę**
        if self.frozen:
            frame = self.frozen_frame.copy()
            bboxes_to_draw = self.frozen_bboxes
        else:
            bboxes_to_draw = self.bboxes

        # Rysowanie bounding boxów na odpowiedniej klatce
        for bbox in bboxes_to_draw:
            x1, y1, x2, y2 = bbox["bbox"]
            color = bbox["color"]
            # Zwiększono grubość ramki z 2 na 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            # Dodaj etykiety dla każdego wykrytego obiektu
            cv2.putText(frame, bbox["id"], (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

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
            # Jeśli pokazujemy preprocessed, przełącz na oryginalny z boxami
            if hasattr(self, 'original_with_boxes') and self.original_with_boxes is not None:
                self.show_frame(self.original_with_boxes, "Oryginalny obraz z detekcją")
                self.show_preprocessing_btn.setText("Pokaż preprocessing")
            else:
                # Jeśli nie mamy zapisanego oryginału z boxami, użyj bieżącej klatki
                if self.original_frame is not None:
                    # Narysuj boxy na kopii oryginalnego obrazu
                    display_frame = self.original_frame.copy()
                    for bbox in self.bboxes:
                        x1, y1, x2, y2 = bbox["bbox"]
                        color = bbox["color"]
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 4)
                        cv2.putText(display_frame, bbox["id"], (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                    self.show_frame(display_frame, "Oryginalny obraz z detekcją")
                else:
                    self.show_frame(self.frozen_frame, "Obraz z detekcją")
                self.show_preprocessing_btn.setText("Pokaż preprocessing")
        else:
            # Jeśli pokazujemy oryginalny, przełącz na preprocessed
            # Najpierw zachowaj aktualny obraz z boxami
            if self.original_frame is not None:
                self.original_with_boxes = self.frozen_frame.copy()
            
            # Wyświetl obraz po preprocessingu z boxami
            preprocessed_with_boxes = self.preprocessed_frame.copy()
            for bbox in self.bboxes:
                x1, y1, x2, y2 = bbox["bbox"]
                color = bbox["color"]
                cv2.rectangle(preprocessed_with_boxes, (x1, y1), (x2, y2), color, 4)
                cv2.putText(preprocessed_with_boxes, bbox["id"], (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                            
            self.show_frame(preprocessed_with_boxes, "Preprocessed z detekcją")
            self.show_preprocessing_btn.setText("Pokaż oryginalny")
            
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
            if self.cap is not None and self.cap.isOpened():
                # Jeśli mamy aktywną kamerę, użyj update_frame
                self.update_frame()  # Odśwież kamerę
            elif self.frozen and self.frozen_frame is not None:
                # Jeśli mamy statyczny obraz, aktualizujemy go ręcznie
                # Pobierz oryginalną ramkę bez boxów
                display_frame = self.frozen_frame.copy()
                
                # Aktualizujemy frozen_bboxes
                self.frozen_bboxes = self.bboxes.copy()
                
                # Rysowanie bounding boxów na obrazie
                for bbox in self.frozen_bboxes:
                    x1, y1, x2, y2 = bbox["bbox"]
                    color = bbox["color"]
                    # Zwiększono grubość ramki z 2 na 4
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 4)
                    # Zwiększono wielkość fonta z 0.5 na 1.0 i grubość z 2 na 3
                    cv2.putText(display_frame, bbox["id"], (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                
                # Wyświetl zaktualizowany obraz
                self.show_frame(display_frame)
                print("Zaktualizowano wyświetlanie statycznego obrazu")
                
            self.cap_label.repaint()  # Wymuś ponowne narysowanie

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()