from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
import cv2
import os
from PyQt5.QtWidgets import QDialog, QLabel, QMessageBox, QFileDialog, QListWidget, QPushButton, QInputDialog, QApplication, QWidget
import torchvision
import torch
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt, QRect, QPropertyAnimation, QPoint, QSequentialAnimationGroup
from models.faster_rcnn import get_model
import urllib.request
import numpy as np
import sys
import matplotlib.pyplot as plt
from io import BytesIO
import csv
import mysql.connector
from datetime import datetime
from db_config import DB_CONFIG
import concurrent.futures
import time
from datetime import datetime
import easyocr
from .ocr_components import ComponentProcessing


class LoadingOverlay(QWidget):
    """Widżet nakładki z animowanym wskaźnikiem ładowania"""
    
    def __init__(self, parent=None):
        super(LoadingOverlay, self).__init__(parent)
        
        # Ustawienia wyglądu
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(parent.size())
        
        # Utworzenie etykiety dla tekstu ładowania
        self.loading_label = QLabel(self)
        self.loading_label.setText("Analizowanie komponentów...")
        self.loading_label.setStyleSheet("""
            color: white;
            background-color: #4b7bec;
            border-radius: 10px;
            padding: 10px 15px;
            font-weight: bold;
        """)
        self.loading_label.setAlignment(Qt.AlignCenter)
        
        # Wyśrodkuj etykietę
        self.loading_label.adjustSize()
        self.loading_label.move(
            self.width() // 2 - self.loading_label.width() // 2,
            self.height() // 2 - self.loading_label.height() // 2
        )
        
        # Wskaźniki kropek dla animacji
        self.dots = [".", "..", "..."]
        self.dots_index = 0
        
        # Timer do animacji kropek
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(500)  # Co 500ms
        
        # Animacja pulsowania etykiety
        self.pulse_animation = QPropertyAnimation(self.loading_label, b"geometry")
        self.pulse_animation.setDuration(1000)
        
        # Ustawienie animacji pulsowania
        start_rect = self.loading_label.geometry()
        end_rect = QRect(
            start_rect.x() - 5, 
            start_rect.y() - 5, 
            start_rect.width() + 10, 
            start_rect.height() + 10
        )
        
        # Utwórz grupę animacji dla efektu pulsowania
        self.animation_group = QSequentialAnimationGroup(self)
        
        # Dodaj animację powiększania
        self.pulse_animation.setStartValue(start_rect)
        self.pulse_animation.setEndValue(end_rect)
        self.animation_group.addAnimation(self.pulse_animation)
        
        # Dodaj animację zmniejszania
        self.pulse_back = QPropertyAnimation(self.loading_label, b"geometry")
        self.pulse_back.setDuration(1000)
        self.pulse_back.setStartValue(end_rect)
        self.pulse_back.setEndValue(start_rect)
        self.animation_group.addAnimation(self.pulse_back)
        
        # Ustaw zapętlenie animacji
        self.animation_group.setLoopCount(-1)  # -1 oznacza zapętlenie w nieskończoność
        self.animation_group.start()
    
    def update_animation(self):
        """Aktualizuje animację kropek"""
        self.dots_index = (self.dots_index + 1) % len(self.dots)
        text = f"Analizowanie komponentów{self.dots[self.dots_index]}"
        self.loading_label.setText(text)
    
    def paintEvent(self, event):
        """Rysuje półprzezroczyste tło"""
        import PyQt5.QtGui as QtGui
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 128))  # Półprzezroczyste czarne tło
    
    def showEvent(self, event):
        """Dopasowanie rozmiaru etykiety podczas pokazywania"""
        super(LoadingOverlay, self).showEvent(event)
        self.loading_label.adjustSize()
        self.loading_label.move(
            self.width() // 2 - self.loading_label.width() // 2,
            self.height() // 2 - self.loading_label.height() // 2
        )

class DetectionWorker(QThread):
    """Klasa do wykonywania detekcji w osobnym wątku"""
    finished = pyqtSignal(object, str)  # Sygnał wysyłany po zakończeniu detekcji (wyniki, nazwa_komponentu)
    
    def __init__(self, model, frame, component_name, confidence_threshold, device, output_dir, component_counter):
        super().__init__()
        self.model = model
        self.frame = frame
        self.component_name = component_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.output_dir = output_dir
        self.component_counter = component_counter
        
    def run(self):
        try:
            # Konwersja BGR do RGB
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            
            # Konwersja do tensora
            tensor_frame = torch.tensor(rgb_frame).permute(2, 0, 1).float() / 255.0
            tensor_frame = tensor_frame.unsqueeze(0).to(self.device)
            
            # Wykonanie detekcji
            with torch.no_grad():
                predictions = self.model(tensor_frame)[0]
            
            # Filtrowanie detekcji na podstawie progu pewności
            detections = []
            count = 0
            
            for box, score, label in zip(predictions["boxes"], predictions["scores"], predictions["labels"]):
                if score > self.confidence_threshold:
                    count += 1
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    component_id = f"{self.component_name}_{count}"
                    detections.append({
                        "id": component_id, 
                        "bbox": (x1, y1, x2, y2), 
                        "score": float(score.item()),
                        "component_type": self.component_name
                    })

                    # Zapisz wycinek komponentu
                    crop = self.frame[y1:y2, x1:x2]
                    crop_filename = os.path.join(self.output_dir, f"{component_id}.png")
                    cv2.imwrite(crop_filename, crop)
            
            # Emituj sygnał z wynikami
            self.finished.emit(detections, self.component_name)
            
        except Exception as e:
            print(f"Błąd w DetectionWorker dla {self.component_name}: {e}")
            import traceback
            traceback.print_exc()
            self.finished.emit([], self.component_name)

class Camera(QDialog):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loadUi("ui/Camera.ui", self)
        self.setGeometry(100, 100, 1200, 800)
        self.cap_label = self.findChild(QLabel, "cap")
        self.bboxes = []  # Inicjalizacja listy wykrytych obiektów
        self.component_list.itemClicked.connect(self.highlight_bbox)
        self.component_counter = 1

        
        # Podłączenie przycisku zapisu
        self.save_button.clicked.connect(self.save_pcb_data)
        
        # Inicjalizacja bazy danych
        self.init_database()
        
        self.frozen = False  # Dodaj tę linię
        self.frozen_frame = None  # Przechowa zamrożoną klatkę
        self.original_frame = None  # Przechowa oryginalną kopię obrazu bez boxów
        self.preprocessed_frame = None  # Przechowa obraz po preprocessingu
        self.is_preprocessed = False  # Flaga czy obraz został już przetworzony
        self.pcb_contour = None  # Przechowuje kontur płytki PCB
        self.pcb_corners = None  # Przechowuje rogi płytki PCB
        self.pos_putted_on = False  # Flaga czy pozycje zostały nałożone na obraz
        
        self.model_paths = {
            "Kondensator": os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/trained/capacitors.pth")),
            "Uklad scalony": os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/trained/ic.pth")),
            "Dioda": os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/trained/dioda_model_epoch_14_mAP_0.822.pth")),
            "USB": os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/trained/usb.pth")),
            "Rezonator": os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/trained/resonator.pth")),
            "Rezystor": os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/trained/resistor.pth")),
            "Przycisk": os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/trained/switch.pth")),
            "Zlacze": os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/trained/connectors.pth")),
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
        self.frame_count = 0

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.analyze_button.clicked.connect(self.toggle_analysis)
        self.virtual_cam_button.clicked.connect(self.choose_virtual_camera)
        self.clear_image_button.clicked.connect(self.clear_image)
        self.pos_file.clicked.connect(self.on_pos_file_click)
        self.mirror_button.clicked.connect(self.toggle_mirror)
        self.comparision_button.clicked.connect(self.on_comparision_click)
        



        
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

        self.is_mirrored = False  # Flaga do śledzenia stanu odbicia lustrzanego

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

        # Nowa lista do przechowywania wszystkich detekcji ze wszystkich modeli
        self.all_detections = []
        
        # Podłączam nowy przycisk do analizy wszystkich komponentów
        self.analyze_all_button = self.findChild(QPushButton, "analyze_all_button")
        self.analyze_all_button.clicked.connect(self.analyze_all_components)
        
        # Stan do śledzenia aktualnie załadowanego modelu
        self.current_model_name = None

    def init_database(self):
        """Inicjalizacja połączenia z bazą danych MySQL"""
        try:
            self.conn = mysql.connector.connect(**DB_CONFIG)
            self.cursor = self.conn.cursor()
            
            # Sprawdź czy tabele istnieją, jeśli nie - utwórz je
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS pcb_records (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    pcb_code VARCHAR(50) UNIQUE,
                    date_analyzed DATETIME,
                    image_path VARCHAR(255)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
            """)
            
            # Sprawdź czy kolumna component_type istnieje, jeśli nie - dodaj ją
            try:
                self.cursor.execute("""
                    ALTER TABLE components ADD COLUMN component_type VARCHAR(100)
                """)
                self.conn.commit()
                print("Dodano kolumnę component_type do tabeli components")
            except mysql.connector.Error as e:
                # Jeśli błąd to "Duplicate column name", to wszystko OK
                if "Duplicate column" not in str(e):
                    print(f"Błąd przy próbie dodania kolumny component_type: {e}")
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS components (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    pcb_code VARCHAR(50),
                    component_id VARCHAR(100),
                    component_type VARCHAR(100),
                    score FLOAT,
                    bbox TEXT,
                    FOREIGN KEY (pcb_code) REFERENCES pcb_records(pcb_code)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
            """)
            
            self.conn.commit()
        except mysql.connector.Error as e:
            print(f"Błąd podczas inicjalizacji bazy danych: {e}")
            QMessageBox.critical(self, "Błąd", f"Nie udało się zainicjalizować bazy danych: {e}")

    def save_pcb_data(self):
        """Zapisuje dane PCB do bazy danych MySQL"""
        if not self.bboxes:
            QMessageBox.warning(self, "Uwaga", "Brak danych do zapisania!")
            return
            
        # Pobierz kod PCB od użytkownika
        pcb_code, ok = QInputDialog.getText(
            self, 'Kod PCB', 
            'Wprowadź kod PCB:',
            text=f"PCB-{datetime.now().strftime('%Y%m%d')}-{len(self.bboxes)}"
        )
        
        if not ok or not pcb_code:
            return
            
        try:
            # Zapisz obraz
            image_dir = "saved_images"
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
                
            image_path = os.path.join(image_dir, f"{pcb_code}.jpg")
            cv2.imwrite(image_path, self.frozen_frame)
            
            # Zapisz dane PCB (jeśli nie istnieje lub aktualizuj)
            self.cursor.execute('''
                INSERT INTO pcb_records (pcb_code, date_analyzed, image_path)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE date_analyzed=%s, image_path=%s
            ''', (pcb_code, datetime.now(), image_path, datetime.now(), image_path))
            
            # Zapisz komponenty
            component_type = self.component.currentText()
            for bbox in self.bboxes:
                self.cursor.execute('''
                    INSERT INTO components (pcb_code, component_id, component_type, score, bbox)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (
                    pcb_code,
                    bbox["id"],
                    component_type,
                    bbox["score"],
                    str(bbox["bbox"])
                ))
            
            self.conn.commit()
            QMessageBox.information(self, "Sukces", f"Dane PCB {pcb_code} zostały zapisane!")
            
            # Odśwież historię jeśli jest otwarta
            for widget in QApplication.topLevelWidgets():
                if widget.__class__.__name__ == "History":
                    widget.load_pcb_data()
                    print("Odświeżono historię po zapisie")
            
        except mysql.connector.IntegrityError:
            QMessageBox.warning(self, "Błąd", "PCB o takim kodzie już istnieje!")
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Wystąpił błąd podczas zapisywania: {e}")
            self.conn.rollback()

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

        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        top = (target_height - new_height) // 2
        bottom = (target_height - new_height + 1) // 2
        left = (target_width - new_width) // 2
        right = (target_width - new_width + 1) // 2

        result = cv2.copyMakeBorder(
            resized_frame,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        return result, left, top  # <--- Zwracasz też marginesy


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
            # Jeśli mamy nałożone markery POS, użyj overlayed_frame, w przeciwnym przypadku użyj preprocessed_frame
            if hasattr(self, 'overlayed_frame') and self.overlayed_frame is not None:
                analyze_frame = self.overlayed_frame.copy()
                print("Używam obrazu z nałożonymi markerami POS")
            else:
                analyze_frame = self.preprocessed_frame.copy()
                print("Używam obrazu po preprocessingu (bez markerów POS)")
                
            print(f"Analizuję obraz o wymiarach: {analyze_frame.shape}")
                
            # Wykonaj analizę na obrazie
            result_frame, detections = self.detect_components(analyze_frame)
            self.update_component_list(detections)
            
            # Rysowanie bounding boxów na obrazie
            # Bazowy obraz do wyświetlenia - ten sam, który użyliśmy do analizy
            if hasattr(self, 'overlayed_frame') and self.overlayed_frame is not None:
                display_frame = self.overlayed_frame.copy()
            else:
                display_frame = self.preprocessed_frame.copy()
                
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
            self.show_frame(display_frame, "")
            print(f"Wykryto {len(self.bboxes)} obiektów")
            
            # Zapisz wyniki detekcji wraz z obrazem
            self.detection_result = {
                "image": display_frame.copy(),
                "detections": self.bboxes.copy()
            }

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
        component_name = self.component.currentText()
        confidence_threshold = self.confidence_thresholds.get(component_name, 0.9)
        print(f"Liczba wykrytych obiektów przed filtrowaniem: {len(predictions['boxes'])}")

                # Zapisz wszystko do jednego folderu
        base_output_dir = "output_components"
        os.makedirs(base_output_dir, exist_ok=True)


        for idx, (box, score, label) in enumerate(zip(predictions["boxes"], predictions["scores"], predictions["labels"])):
            if score > confidence_threshold:
                count += 1
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                component_id = f"{component_name}"
                detections.append({
                    "id": component_id,
                    "bbox": (x1, y1, x2, y2),
                    "score": float(score.item())
                })

                # Zapisz wycinek komponentu
                crop = frame[y1:y2, x1:x2]
                crop_filename = os.path.join(base_output_dir, f"{component_id}.png")
                cv2.imwrite(crop_filename, crop)

        self.count_elements.setText(f"{count}")
        print(f"Liczba wykrytych obiektów po filtrowaniu (próg={confidence_threshold}): {count}")
        print(f"Wycinki zapisano do folderu: {base_output_dir}")

        # Dodaj prostokąty do obrazu
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
            self.show_frame(processed_img, "")
            
            # Aktywuj przyciski
            self.analyze_button.setEnabled(True)
            self.show_preprocessing_btn.setEnabled(True)
            
            # Ustaw tryb wyświetlania na preprocessed
            self.preprocessing_visible = True
            self.show_preprocessing_btn.setText("Pokaż detekcję")
            
            QMessageBox.information(self, "Informacja", "Płytka PCB została wycięta i przetworzona. Możesz teraz możesz nałożyć POS.")
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
            corners = corners.astype(int)
        
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
        box = box.astype(int)
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
        self.show_frame(rotated_pcb, "")
        
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
        self.show_frame(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "")
        
        # 3. Zastosuj operacje morfologiczne, aby usunąć szum i wzmocnić kontury
        kernel = np.ones((15, 15), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        
        # Pokaż obraz po operacjach morfologicznych
        self.show_frame(cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR), "")
        
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
            box = box.astype(int)
            
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
                self.show_frame(rotated_pcb, "")
                
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
            box = box.astype(int)
            
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
            self.show_frame(rotated_pcb, "")
            
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
        Delikatny preprocessing obrazu dla systemów AOI.
        0) Wykrycie i wycięcie płytki PCB
        1) Korekcja optyczna dystorsji (jeśli dostępna)
        2) Delikatne odszumianie
        3) Delikatna poprawa kontrastu
        4) Bardzo subtelne wyrównanie histogramu
        5) Rejestracja obrazu za pomocą fiduciali (jeśli dostępna)
        6) Wstępna segmentacja (HSV i progi)

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
        
        # 1. Korekcja optyczna (dystorsji) - bez zmian, to potrzebna korekcja
        if hasattr(self, 'camera_matrix') and hasattr(self, 'dist_coeffs'):
            pcb_image = cv2.undistort(pcb_image, self.camera_matrix, self.dist_coeffs)
            
        # 2. Delikatne odszumianie - nieco silniejsze niż poprzednio, ale wciąż subtelne
        # Używamy łagodnego filtra bilateralnego
        denoised = cv2.bilateralFilter(pcb_image, d=3, sigmaColor=15, sigmaSpace=15)
        
        # Pokaż obraz po odszumianiu
        self.show_frame(denoised, "")
        
        # 3. Delikatna poprawa kontrastu i jasności
        adjusted = cv2.convertScaleAbs(denoised, alpha=1.05, beta=3)
        
        # Pokaż obraz po delikatnej poprawie kontrastu
        self.show_frame(adjusted, "")
        
        # 4. Bardzo subtelne wyrównanie histogramu tylko na kanale jasności
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Bardzo delikatne CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8,8))
        v = clahe.apply(v)
        hsv = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Pokaż obraz po delikatnym wyrównaniu histogramu
        self.show_frame(result, "")

        # 5. Rejestracja obrazu (alignment) - fiduciale (bez zmian, to potrzebna funkcjonalność)
        if hasattr(self, 'fiducial_detector'):
            pts_src = self.fiducial_detector.detect(pcb_image)
            pts_dst = self.fiducial_detector.reference_points
            if len(pts_src) >= 3:
                M, _ = cv2.estimateAffinePartial2D(np.array(pts_src), np.array(pts_dst))
                result = cv2.warpAffine(result, M, (result.shape[1], result.shape[0]))

        # 6. Wstępna segmentacja (bez zmian, to tylko analityczne maski)
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
        Wykonuje delikatny preprocessing obrazu PCB bez etapu wykrywania płytki.
        Ten obraz powinien być już przyciętą płytką PCB.
        
        1) Korekcja optyczna dystorsji (jeśli dostępna)
        2) Delikatne odszumianie
        3) Delikatna poprawa kontrastu
        4) Bardzo subtelne wyrównanie histogramu
        5) Rejestracja obrazu za pomocą fiduciali (jeśli dostępna)
        6) Wstępna segmentacja (HSV i progi)

        Zwraca przetworzony obraz BGR, a maski segmentacji są w self.masks.
        """
        # Zapisz oryginalne wymiary obrazu z płytką do późniejszego odniesienia
        self.pcb_image_original = image.copy()
        
        # 1. Korekcja optyczna (dystorsji) - bez zmian, to potrzebna korekcja
        if hasattr(self, 'camera_matrix') and hasattr(self, 'dist_coeffs'):
            image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
            
        # 2. Delikatne odszumianie - nieco silniejsze niż poprzednio, ale wciąż subtelne
        # Używamy łagodnego filtra bilateralnego
        denoised = cv2.bilateralFilter(image, d=3, sigmaColor=15, sigmaSpace=15)
        
        # Pokaż obraz po odszumianiu
        self.show_frame(denoised, "")
        
        # 3. Delikatna poprawa kontrastu i jasności
        adjusted = cv2.convertScaleAbs(denoised, alpha=1.05, beta=3)
        
        # Pokaż obraz po delikatnej poprawie kontrastu
        self.show_frame(adjusted, "")
        
        # 4. Bardzo subtelne wyrównanie histogramu tylko na kanale jasności
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Bardzo delikatne CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8,8))
        v = clahe.apply(v)
        hsv = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Pokaż obraz po delikatnym wyrównaniu histogramu
        self.show_frame(result, "")

        # 5. Rejestracja obrazu (alignment) - fiduciale (bez zmian, to potrzebna funkcjonalność)
        if hasattr(self, 'fiducial_detector'):
            pts_src = self.fiducial_detector.detect(image)
            pts_dst = self.fiducial_detector.reference_points
            if len(pts_src) >= 3:
                M, _ = cv2.estimateAffinePartial2D(np.array(pts_src), np.array(pts_dst))
                result = cv2.warpAffine(result, M, (result.shape[1], result.shape[0]))

        # 6. Wstępna segmentacja (bez zmian, to tylko analityczne maski)
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

    def load_and_preprocess_image(self, image):
        """
        Wczytuje obraz, oblicza marginesy oraz rozmiar obrazu w oknie.
        Funkcja nie wykonuje preprocessingu.
        """
        # 1. Uzyskaj rozmiary obrazu
        original_height, original_width, _ = image.shape

        # 2. Resize do rozmiaru labelki (z zachowaniem proporcji)
        resized_frame, margin_x, margin_y = self.resize_with_aspect_ratio(
            image,  # używamy oryginalnego obrazu, bez preprocessingu
            self.cap_label.width(),  # szerokość labelki
            self.cap_label.height()  # wysokość labelki
        )

        # 3. Zapisz rozmiar obrazu w oknie i marginesy jako atrybuty obiektu
        self.resized_frame = resized_frame
        self.margin_x = margin_x
        self.margin_y = margin_y

    def overlay_pos_markers(self, preprocessed_frame, pos_file_path, target_width, target_height, pcb_width_mm=43.18, pcb_height_mm=17.78):
        """
        Nakłada punkty z pliku POS na obraz płytki PCB.
        :param preprocessed_frame: obraz płytki (przed resize)
        :param pos_file_path: ścieżka do pliku POS (.csv)
        :param pcb_width_mm: fizyczna szerokość płytki w mm
        :param pcb_height_mm: fizyczna wysokość płytki w mm
        :return: obraz z nałożonymi punktami
        """
        # Zapisz ścieżkę do pliku POS dla ponownego nakładania przy zmianie odbicia
        self.pos_file_path = pos_file_path
        
        # Wywołanie preprocessing i resize
        self.load_and_preprocess_image(preprocessed_frame)

        print("Rozpoczynam nakładanie markerów POS...")
        print(f"Rozmiar płytki: {pcb_width_mm} mm x {pcb_height_mm} mm")
        print(f"Rozmiar okna: {target_width} x {target_height} pikseli")

        # Skala w pikselach na mm, ale biorąc pod uwagę rozmiar okna
        resized_h, resized_w, _ = self.preprocessed_frame.shape
        scale_x = resized_w / pcb_width_mm  # Piksele na mm (szerokość)
        scale_y = resized_h / pcb_height_mm  # Piksele na mm (wysokość)

        print(f"Skala: {scale_x} x {scale_y}")
        print(f"Marginesy: {self.margin_x} x {self.margin_y}")

        # Kopia obrazu do rysowania
        result = self.preprocessed_frame.copy()

        # Czytaj plik POS
        with open(pos_file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Współrzędne w mm
                try:
                    pos_x_mm = float(row["PosX"]) - 126.873
                    pos_y_mm = float(row["PosY"]) + 96.114
                    ref = row["Ref"]
                    print(f"Wartości z POS: {ref} ({pos_x_mm}, {pos_y_mm})")
                except (ValueError, KeyError):
                    continue  # Pomiń błędne wpisy

                # Przekształcenie współrzędnych POS na współrzędne w pikselach
                # Uwzględnienie odbicia lustrzanego dla współrzędnej X jeśli aktywne
                if self.is_mirrored:
                    pixel_x = int((pcb_width_mm - pos_x_mm) * scale_x)  # Odbicie w poziomie
                    print(f"Stosowanie odbicia lustrzanego dla POS {ref}")
                else:
                    pixel_x = int(pos_x_mm * scale_x)
                
                pixel_y = int(-(pos_y_mm) * scale_y)     # Odwracamy oś Y
                print(f"Współrzędne w pikselach dla {ref}: ({pixel_x}, {pixel_y})")

                # Rysowanie punktu na obrazie
                cv2.circle(result, (pixel_x, pixel_y), 8, (0, 0, 255), -1)

                # Powiększenie rozmiaru bbox o 2.5 razy
                bbox_size = 12 * 2.5  # Powiększenie rozmiaru o 2.5 razy
                bbox_half_size = int(bbox_size / 2)

                # Rysowanie powiększonego bounding boxa wokół punktu
                cv2.rectangle(result, 
                              (pixel_x - bbox_half_size, pixel_y - bbox_half_size), 
                              (pixel_x + bbox_half_size, pixel_y + bbox_half_size), 
                              (0, 255, 0), 2)  # Bbox w kolorze zielonym

                # Opcjonalnie: Rysowanie napisu 'Ref' obok punktu
                cv2.putText(result, ref, (pixel_x + 8, pixel_y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        self.pos_putted_on = True  # Ustaw flagę, że POS został nałożony

        self.overlayed_frame = result.copy()
        self.show_frame(result, "Obraz z nałożonymi markerami POS")

    
    def bbox_matches_pos(self, preprocessed_frame, pos_file_path, target_width, target_height, pcb_width_mm=43.18, pcb_height_mm=17.78):
        """
        Nakłada same bboxy z pliku POS na obraz płytki PCB i zapisuje je do self.pos_bboxes.
        """
        # 🧩 Mapowanie nazw z POS na klasy modelowe
        component_name_map = {
            "J1": "Zlacze",
            "J2": "Zlacze",
            "J3": "Zlacze",
            "J4": "Zlacze",
            "SW1": "Przycisk",
            "IC3": "Uklad scalony",
            "U$4": "Uklad scalony",
            "U$5": "Uklad scalony",
            "Y1": "Rezonator",
            "D+0": "USB",
            "D-0": "USB",
            "RX0": "USB",
            "TX0": "USB",
            "L0": "Rezystor",
            "PWR0": "Rezystor",
            "FRAME1": "Rezystor",
            "UNK_HOLE_0": "Zlacze",
            "UNK_HOLE_1": "Zlacze",
            "UNK_HOLE_2": "Zlacze",
            "UNK_HOLE_3": "Zlacze"
        }

        self.pos_file_path = pos_file_path
        self.load_and_preprocess_image(preprocessed_frame)

        print("Rozpoczynam nakładanie bboxów z POS...")
        print(f"Rozmiar płytki: {pcb_width_mm} mm x {pcb_height_mm} mm")
        print(f"Rozmiar okna: {target_width} x {target_height} pikseli")

        resized_h, resized_w, _ = self.preprocessed_frame.shape
        scale_x = resized_w / pcb_width_mm
        scale_y = resized_h / pcb_height_mm

        print(f"Skala: {scale_x} x {scale_y}")
        print(f"Marginesy: {self.margin_x} x {self.margin_y}")

        result = self.preprocessed_frame.copy()
        self.pos_bboxes = []  # <----- 🆕 Lista bboxów z POS

        with open(pos_file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    pos_x_mm = float(row["PosX"]) - 126.873
                    pos_y_mm = float(row["PosY"]) + 96.114
                    ref = row["Ref"]
                    print(f"Wartości z POS: {ref} ({pos_x_mm}, {pos_y_mm})")
                except (ValueError, KeyError):
                    continue

                # 🧠 Mapowanie ref na kategorię komponentu
                mapped_component = component_name_map.get(ref, ref)

                if self.is_mirrored:
                    pixel_x = int((pcb_width_mm - pos_x_mm) * scale_x)
                    print(f"Stosowanie odbicia lustrzanego dla POS {ref}")
                else:
                    pixel_x = int(pos_x_mm * scale_x)

                pixel_y = int(-(pos_y_mm) * scale_y)
                print(f"Współrzędne w pikselach dla {ref}: ({pixel_x}, {pixel_y})")

                bbox_size = 12 * 2.5
                bbox_half_size = int(bbox_size / 2)

                top_left = (pixel_x - bbox_half_size, pixel_y - bbox_half_size)
                bottom_right = (pixel_x + bbox_half_size, pixel_y + bbox_half_size)

                # 🟩 Rysuj bbox
                cv2.rectangle(result, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(result, ref, (pixel_x + 8, pixel_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

                # 🧠 Zapisz bbox do listy
                self.pos_bboxes.append({
                    'component': mapped_component,
                    'x': top_left[0],
                    'y': top_left[1],
                    'w': bbox_half_size * 2,
                    'h': bbox_half_size * 2
                })

        print(f"POS bboxes załadowane: {[b['component'] for b in self.pos_bboxes]}")

        if not self.frozen or self.preprocessed_frame is None:
            QMessageBox.warning(self, "Uwaga", "Najpierw załaduj zdjęcie i wykonaj preprocessing!")
            return

        if hasattr(self, 'workers') and self.workers:
            QMessageBox.warning(self, "Uwaga", "Analiza jest już w toku!")
            return

        if hasattr(self, 'overlayed_frame') and self.overlayed_frame is not None:
            analyze_frame = self.overlayed_frame.copy()
        else:
            analyze_frame = self.preprocessed_frame.copy()

        self.analyze_all_button.setText("Analizowanie...")
        self.analyze_all_button.setEnabled(False)

        self.loading_overlay = LoadingOverlay(self)
        self.loading_overlay.show()

        components = list(self.model_paths.keys())
        models_to_process = []

        for component_name in components:
            model_path = self.model_paths[component_name]
            confidence_threshold = self.confidence_thresholds.get(component_name, 0.9)

            try:
                model = get_model(2)
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()

                models_to_process.append({
                    'model': model,
                    'component_name': component_name,
                    'confidence_threshold': confidence_threshold
                })

            except Exception as e:
                print(f"Błąd podczas ładowania modelu {component_name}: {e}")
                import traceback
                traceback.print_exc()

        self.all_detections = []
        self.bboxes = []
        self.workers = []

        for model_info in models_to_process:
            output_dir = "output_components"
            os.makedirs(output_dir, exist_ok=True)

            worker = DetectionWorker(
                model_info['model'],
                analyze_frame,
                model_info['component_name'],
                model_info['confidence_threshold'],
                self.device,
                output_dir,
                self.component_counter,
            )
            worker.finished.connect(self.handle_detection_results)
            self.workers.append(worker)
            worker.start()

        self.check_workers_timer = QTimer()
        self.check_workers_timer.timeout.connect(self.check_workers_compare)
        self.check_workers_timer.start(1000)

        self.pos_putted_on = True
        self.overlayed_frame = result.copy()
        self.show_frame(result, "Obraz z nałożonymi bboxami POS")


    def porownaj_bboxy(self):
        if not hasattr(self, 'model_bboxes') or not hasattr(self, 'pos_bboxes'):
            QMessageBox.warning(self, "Błąd", "Brakuje danych do porównania.")
            return

        name_map = {
            "usb": "USB",
            "rezystor": "Rezystor",
            "kondensator": "Kondensator",
            "zlacze": "Zlacze",
            "uklad scalony": "Uklad scalony",
            "przycisk": "Przycisk",
            "rezonator": "Rezonator",
        }

        dopasowane = []
        niedopasowane_model = []
        niedopasowane_pos = self.pos_bboxes.copy()

        def point_in_bbox(point, bbox):
            px, py = point
            bx, by, bw, bh = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            return bx <= px <= bx + bw and by <= py <= by + bh

        for model_box in self.model_bboxes:
            matched = False
            model_name = model_box['component'].strip().lower()

            for pos_box in self.pos_bboxes:
                pos_name = pos_box['component'].strip().lower()
                mapped_pos_name = name_map.get(pos_name, pos_name)

                if model_name != mapped_pos_name.lower():
                    continue

                # Obliczamy środek POS bbox
                pos_center = (pos_box['x'] + pos_box['w'] / 2, pos_box['y'] + pos_box['h'] / 2)

                if point_in_bbox(pos_center, model_box):
                    dopasowane.append((model_box, pos_box))
                    if pos_box in niedopasowane_pos:
                        niedopasowane_pos.remove(pos_box)
                    matched = True
                    break

            if not matched:
                niedopasowane_model.append(model_box)

        # Wyniki:
        print(f"Dopasowane bboxy ({len(dopasowane)}):")
        for i, (model_box, pos_box) in enumerate(dopasowane):
            print(f"{i+1}. Model={model_box['component']} | POS={pos_box['component']}")

        print(f"\nNiedopasowane bboxy z modelu ({len(niedopasowane_model)}):")
        for box in niedopasowane_model:
            print(f"- {box['component']}")

        print(f"\nNiedopasowane bboxy z POS ({len(niedopasowane_pos)}):")
        for box in niedopasowane_pos:
            print(f"- {box['component']}")

        QMessageBox.information(self, "Porównanie bboxów",
            f"Dopasowania: {len(dopasowane)}\n"
            f"Niedopasowane (model): {len(niedopasowane_model)}\n"
            f"Niedopasowane (POS): {len(niedopasowane_pos)}"
        )

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
            cv2.putText(frame, "",
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Wyświetlanie obrazu za pomocą metody show_frame
        self.show_frame(frame)

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
        
        # Usuwamy odbicie lustrzane całego obrazu - nie stosujemy go już tutaj
        # if self.is_mirrored:
        #     display_frame = cv2.flip(display_frame, 1)  # 1 oznacza odbicie w poziomie (wokół osi Y)
        
        # Używamy resize_with_aspect_ratio zamiast zwykłego resize
        # aby zachować proporcje obrazu
        frame_resized,margin_x,margin_y = self.resize_with_aspect_ratio(
            display_frame, 
            self.cap_label.width(), 
            self.cap_label.height()
        )
        
        h, w, ch = frame_resized.shape
        bytes_per_line = ch * w
        
        # Zapewniamy poprawny format koloru (BGR dla OpenCV)
        qimg = QImage(frame_resized.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.cap_label.setPixmap(QPixmap.fromImage(qimg))

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
                self.show_frame(display_frame, "")
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
                self.show_frame(display_frame, "")
                self.show_preprocessing_btn.setText("Pokaż preprocessing")
        else:
            # Jeśli pokazujemy obraz z boxami, przełącz na czysty preprocessed
            preprocessed_view = self.preprocessed_frame.copy()
            self.show_frame(preprocessed_view, "")
            self.show_preprocessing_btn.setText("Pokaż detekcję")
            
        self.preprocessing_visible = not self.preprocessing_visible

    def show_frozen_frame(self):
        """Wyświetla ostatnią zamrożoną klatkę"""
        if self.frozen_frame is not None:
            self.show_frame(self.frozen_frame)

    def clear_image(self):
        """Resetuje aplikację do stanu początkowego - czyści wszystkie obrazy, boxy i dane"""
        # Resetowanie stanu kamery
        self.frozen = False
        self.frozen_frame = None
        self.original_frame = None
        self.preprocessed_frame = None
        self.is_preprocessed = False
        
        # Czyszczenie wykrytych obiektów
        self.bboxes = []  
        self.component_list.clear()  
        self.count_elements.setText("")
        
        # Czyszczenie markerów POS
        self.pos_putted_on = False
        if hasattr(self, 'overlayed_frame'):
            self.overlayed_frame = None
        if hasattr(self, 'pos_file_path'):
            self.pos_file_path = None
            
        # Resetowanie masek
        if hasattr(self, 'masks'):
            self.masks = {}
            
        # Czyszczenie wyników detekcji
        if hasattr(self, 'detection_result'):
            self.detection_result = None
            
        # Czyszczenie konturów PCB
        self.pcb_contour = None
        self.pcb_corners = None
        
        # Resetowanie UI
        self.cap_label.clear()
        
        # Resetowanie analizy
        if self.analyze:
            self.analyze = False
            self.analyze_button.setText("Analiza")
            
        # Dezaktywacja przycisków, które wymagają załadowanego obrazu
        self.analyze_button.setEnabled(False)
        self.show_preprocessing_btn.setEnabled(False)
        self.preprocessing_btn.setEnabled(False)
        
        # Informacja dla użytkownika
        print("Wszystkie dane zostały wyczyszczone, aplikacja zresetowana do stanu początkowego")

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
        
        # Aktywuj przycisk zapisu jeśli są wykryte komponenty
        self.save_button.setEnabled(len(self.bboxes) > 0)

    def highlight_bbox(self, item):
        """Podświetla wybrany bounding box na obrazie"""
        if not self.frozen_frame is None:
            # Kopiuj oryginalną ramkę
            display_frame = self.frozen_frame.copy()
            
            # Znajdź wybrany element
            selected_id = item.text()
            
            # Znajdź odpowiadający box
            selected_bbox = None
            for bbox in self.frozen_bboxes:
                if bbox["id"] == selected_id:
                    selected_bbox = bbox
                    break
            
            if selected_bbox:
                # Rysuj wszystkie boxy, ale utwórz specjalny styl dla wybranego
                for bbox in self.frozen_bboxes:
                    x1, y1, x2, y2 = bbox["bbox"]
                    
                    # Określ kolor i grubość linii
                    if bbox["id"] == selected_id:
                        color = (0, 0, 255)  # Czerwony dla wybranego
                        thickness = 4
                    else:
                        color = bbox["color"]
                        thickness = 2
                    
                    # Rysuj prostokąt
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Rysuj etykietę
                    cv2.putText(display_frame, bbox["id"], (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            
            # Wyświetl ramkę z podświetlonym boxem
            self.show_frame(display_frame, "")

    def closeEvent(self, event):
        """Zamknięcie okna"""
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
            
        super().closeEvent(event)
        
    def toggle_mirror(self):
        """Przełącza stan odbicia lustrzanego obrazu"""
        self.is_mirrored = not self.is_mirrored
        
        # Zastosuj zmiany do aktualnie wyświetlanego obrazu
        if self.frozen:
            if self.frozen_frame is not None:
                if self.is_mirrored:
                    self.frozen_frame = cv2.flip(self.frozen_frame, 1)  # 1 oznacza odbicie poziome
                self.show_frame(self.frozen_frame, "")
                
        # Aktualizuj stan przycisku
        if self.is_mirrored:
            self.mirror_button.setStyleSheet("background-color: #4b7bec; color: white;")
        else:
            self.mirror_button.setStyleSheet("")

    def on_pos_file_click(self):
        """Obsługa kliknięcia przycisku wyboru pliku POS"""
        if not self.frozen or self.preprocessed_frame is None:
            QMessageBox.warning(self, "Uwaga", "Najpierw załaduj zdjęcie i wykonaj preprocessing!")
            return
            
        # Wyświetl dialog wyboru pliku
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Wybierz plik POS", "", 
            "Pliki CSV (*.csv);;Wszystkie pliki (*)", 
            options=options
        )
        
        if not file_name:
            return
            
        try:
            # Nałóż markery POS na obraz
            self.overlay_pos_markers(
                self.preprocessed_frame,
                file_name,
                self.cap_label.width(),
                self.cap_label.height()
            )
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Nie udało się nałożyć markerów POS: {str(e)}")
    
        

    def analyze_all_components(self):
        """Analizuje wszystkie typy komponentów jednocześnie na obrazie"""
        if not self.frozen or self.preprocessed_frame is None:
            QMessageBox.warning(self, "Uwaga", "Najpierw załaduj zdjęcie i wykonaj preprocessing!")
            return
        
        # Jeśli analiza już jest w toku, zatrzymaj ją
        if hasattr(self, 'workers') and self.workers:
            QMessageBox.warning(self, "Uwaga", "Analiza jest już w toku!")
            return
        
        # Pobierz przetworzony obraz
        if hasattr(self, 'overlayed_frame') and self.overlayed_frame is not None:
            analyze_frame = self.overlayed_frame.copy()
        else:
            analyze_frame = self.preprocessed_frame.copy()
        
        # Zmień tekst przycisku i wyłącz go na czas analizy
        self.analyze_all_button.setText("Analizowanie...")
        self.analyze_all_button.setEnabled(False)
        
        # Pokaż nakładkę ładowania
        self.loading_overlay = LoadingOverlay(self)
        self.loading_overlay.show()
        
        # Lista wszystkich modeli do przetworzenia
        components = list(self.model_paths.keys())
        models_to_process = []
        
        for component_name in components:
            model_path = self.model_paths[component_name]
            confidence_threshold = self.confidence_thresholds.get(component_name, 0.9)
            
            # Utwórz model dla tego komponentu
            try:
                model = get_model(2)
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                
                models_to_process.append({
                    'model': model,
                    'component_name': component_name,
                    'confidence_threshold': confidence_threshold
                })
                
            except Exception as e:
                print(f"Błąd podczas ładowania modelu {component_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Wyczyść poprzednie detekcje
        self.all_detections = []
        self.bboxes = []
        
        # Utwórz wątki dla każdego modelu
        self.workers = []
        for model_info in models_to_process:
            output_dir = "output_components"
            os.makedirs(output_dir, exist_ok=True)

            worker = DetectionWorker(
                model_info['model'],
                analyze_frame,
                model_info['component_name'],
                model_info['confidence_threshold'],
                self.device,
                output_dir,
                self.component_counter,
            )
            worker.finished.connect(self.handle_detection_results)
            self.workers.append(worker)
            worker.start()
        
        # Uruchom timer sprawdzający stan wszystkich wątków
        self.check_workers_timer = QTimer()
        self.check_workers_timer.timeout.connect(self.check_workers_compare)
        self.check_workers_timer.start(1000)  # Sprawdzaj co 1 sekundę

    def on_comparision_click(self):
        """Obsługa kliknięcia przycisku wyboru pliku POS"""
        if not self.frozen or self.preprocessed_frame is None:
            QMessageBox.warning(self, "Uwaga", "Najpierw załaduj zdjęcie i wykonaj preprocessing!")
            return
            
        # Wyświetl dialog wyboru pliku
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Wybierz plik POS", "", 
            "Pliki CSV (*.csv);;Wszystkie pliki (*)", 
            options=options
        )
        
        if not file_name:
            return
            
        try:
            # Nałóż markery POS na obraz
            self.bbox_matches_pos(
                self.preprocessed_frame,
                file_name,
                self.cap_label.width(),
                self.cap_label.height()
            )
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Nie udało się nałożyć markerów POS: {str(e)}")

        #self.analyze_all_components()
        
    
    def handle_detection_results(self, detections, component_name):
        """Obsługuje wyniki detekcji z wątku DetectionWorker"""

        if detections:
            print(f"Otrzymano {len(detections)} detekcji dla komponentu {component_name}")

            if not hasattr(self, 'model_bboxes'):
                self.model_bboxes = []

            for det in detections:
                self.all_detections.append(det)

                x1, y1, x2, y2 = det['bbox']
                confidence = det.get('confidence', 0.0)
                width = x2 - x1
                height = y2 - y1

                self.model_bboxes.append({
                    'component': component_name,
                    'x': x1,
                    'y': y1,
                    'w': width,
                    'h': height,
                    'confidence': confidence
                })

                # 🟥 Rysowanie bboxów modelu na overlayed_frame
                if hasattr(self, 'overlayed_frame') and self.overlayed_frame is not None:
                    cv2.rectangle(self.overlayed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(self.overlayed_frame, component_name, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                # Wypisz nazwę komponentu i jego wymiary bboxa
                print(f" - {component_name}: x={x1}, y={y1}, w={width}, h={height}, confidence={confidence:.2f}")

            # Po narysowaniu nowych bboxów odśwież widok:
            self.show_frame(self.overlayed_frame, "Obraz z wykrytymi komponentami (model)")
            print(f"Model bboxes aktualnie zapisane: {[b['component'] for b in self.model_bboxes]}")
        else:
            print(f"Brak detekcji dla komponentu {component_name}")



    
    def check_workers_status(self):
        """Sprawdza stan wszystkich wątków detekcji i aktualizuje UI po zakończeniu"""
        all_finished = all(not worker.isRunning() for worker in self.workers)
        

        if all_finished:
            self.check_workers_timer.stop()
            
            # Ukryj i usuń nakładkę ładowania
            if hasattr(self, 'loading_overlay') and self.loading_overlay:
                self.loading_overlay.animation_timer.stop()
                self.loading_overlay.animation_group.stop()
                self.loading_overlay.hide()
                self.loading_overlay.deleteLater()
                self.loading_overlay = None
            
            # 🔁 Najpierw OCR – musi być przed `process_all_detections()`
            processor = ComponentProcessing(
                input_root="output_components",
                preprocessed_output_root="preprocessed_rot_components",
                output_best_txt="recognized_best_components.txt",
                output_all_txt="recognized_all_rotations.txt"
            )
            processor.process_components()

            # 🧠 Dopiero teraz aktualizuj interfejs (który używa wyników OCR)
            self.process_all_detections()

            # Zresetuj stan
            self.workers = []
            self.analyze_all_button.setText("Analizuj wszystko")
            self.analyze_all_button.setEnabled(True)
    
    def check_workers_compare(self):
        all_done = all(not worker.isRunning() for worker in self.workers)

        if all_done:
            self.check_workers_timer.stop()
            self.analyze_all_button.setEnabled(True)
            self.analyze_all_button.setText("Analizuj wszystkie")

            if hasattr(self, 'loading_overlay'):
                self.loading_overlay.close()

            print("Wszystkie detekcje zakończone.")
            self.model_bboxes = self.bboxes  # 🧠 <- zapisz bboxy modelu do porównania

            # 🆕 Porównaj bboxy z POS i detekcji modelu
            self.porownaj_bboxy()
    
    def process_all_detections(self):
        """Przetwarza wszystkie detekcje i aktualizuje UI"""
        if not self.all_detections:
            QMessageBox.information(self, "Informacja", "Nie wykryto żadnych komponentów!")
            return
        
        # Aktualizuj listę komponentów
        self.update_component_list(self.all_detections)
        
        # Wyświetl obraz z zaznaczonymi komponentami
        if hasattr(self, 'overlayed_frame') and self.overlayed_frame is not None:
            display_frame = self.overlayed_frame.copy()
        else:
            display_frame = self.preprocessed_frame.copy()
        
        # Rysuj bounding boxy dla wszystkich detekcji
        for bbox in self.bboxes:
            x1, y1, x2, y2 = bbox["bbox"]
            color = bbox["color"]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(display_frame, bbox["id"], (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        # Zapamiętaj ramkę z boxami
        self.frozen_frame = display_frame.copy()
        self.frozen_bboxes = self.bboxes.copy()
        
        # Wyświetl zaktualizowany obraz
        self.show_frame(display_frame, "")
        
        # Aktywuj przycisk zapisu
        self.save_button.setEnabled(True)
        
        # Informacja dla użytkownika
        print(f"Łącznie wykryto {len(self.bboxes)} komponentów")
        QMessageBox.information(self, "Informacja", f"Wykryto {len(self.bboxes)} komponentów różnych typów")

    
    def update_component_list(self, detections):
        """Aktualizuje listę ID komponentów na podstawie wykrytych obiektów i wyników OCR"""
        self.component_list.clear()
        self.bboxes = []

        print(f"Otrzymano {len(detections)} detekcji do aktualizacji listy")

        recognized_txt = "recognized_best_components.txt"

        # Wczytaj wyniki OCR do słownika: {"Kondensator_1.png": ("tekst", "90")}
        recognized_map = {}
        if os.path.exists(recognized_txt):
            with open(recognized_txt, "r", encoding="utf-8") as f:
                for line in f:
                    if ": [" in line and "] -> " in line:
                        try:
                            filename_part, rest = line.strip().split(": [")
                            angle_part, text = rest.split("] -> ")
                            recognized_map[filename_part.strip()] = (text.strip(), angle_part.strip("°"))
                        except ValueError:
                            continue

        # Grupowanie OCR po typie komponentu: {"Kondensator": [(text, angle), ...]}
        grouped_ocr = {}
        for filename, (text, angle) in recognized_map.items():
            if "_" in filename:
                component_type = filename.rsplit("_", 1)[0]
                grouped_ocr.setdefault(component_type, []).append((text, angle))

        # Licznik dla każdego typu komponentu
        component_counters = {}

        if isinstance(detections, list) and detections and isinstance(detections[0], dict):
            for detection in detections:
                x1, y1, x2, y2 = map(int, detection["bbox"])
                component_type = detection.get("component_type", "Nieznany")

                index = component_counters.get(component_type, 0)
                ocr_list = grouped_ocr.get(component_type, [])

                if index < len(ocr_list):
                    ocr_text, best_angle = ocr_list[index]
                else:
                    ocr_text, best_angle = "(brak OCR)", "-"

                component_counters[component_type] = index + 1

                display_text = f"{component_type} ({ocr_text}): score {detection['score']:.2f}"
                self.component_list.addItem(display_text)

                self.bboxes.append({
                    "id": f"{component_type}_{index + 1}",
                    "bbox": (x1, y1, x2, y2),
                    "color": (0, 255, 0),
                    "score": detection["score"],
                    "component_type": component_type
                })

                print(f"{component_type}_{index + 1}: OCR -> {ocr_text} (angle: {best_angle}°)")

        else:
            print("Nieznany format detekcji lub pusta lista")

        print(f"Zaktualizowano listę komponentów, dodano {len(self.bboxes)} elementów")
        self.save_button.setEnabled(len(self.bboxes) > 0)





