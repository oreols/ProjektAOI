from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
import cv2
import os
from PyQt5.QtWidgets import QDialog, QLabel, QMessageBox, QFileDialog, QListWidget, QInputDialog, QLineEdit, QComboBox, QPushButton
import torchvision
import torch
from PyQt5.QtCore import QTimer
from models.faster_rcnn import get_model
import urllib.request
import numpy as np
import re




    
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



        self.model_paths = {
            "Kondensator": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/trained_components/best_model_epoch_68_mAP_0.282.pth")),
            "Układ scalony": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/trained_components/ic_resnet50v2_model_epoch_12_mAP_0.648.pth")),
            "Zworka": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/trained_components/jumpers_resnet50v2_model_epoch_49_mAP_0.469.pth")),
            "USB": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/trained_components/usb_resnet50v2_model_epoch_9_mAP_0.799.pth")),
            "Rezonator kwarcowy": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/trained_components/resonator_resnet50v2_model_epoch_23_mAP_0.820.pth")),
            "Rezystor": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/trained_components/best_model_epoch_65_mAP_0.316.pth")),
            "Cewka": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/trained_components/cewka_80_mAP_0.760.pth")),
            "Złącze": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/trained_components/connectors_resnet50v2_model_epoch_58_mAP_0.650.pth")),
            "Tranzystor": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/trained_components/transistors-tactswitches_resnet50v2_model_epoch_21_mAP_0.755.pth")),
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
        self.pcb_color_select = self.findChild(QComboBox, "pcb_color_select")
        self.pcb_width_input = self.findChild(QLineEdit, "pcb_width_input")
        self.pcb_height_input = self.findChild(QLineEdit, "pcb_height_input")
        self.load_pos_button = self.findChild(QPushButton, "load_pos_button")
        self.load_pos_button.clicked.connect(self.load_and_overlay_pos)
        self.pcb_width_input = self.findChild(QLineEdit, "pcb_width_input")
        self.pcb_height_input = self.findChild(QLineEdit, "pcb_height_input")

        self.component.addItem("Kondensator")
        self.component.addItem("Układ scalony")
        self.component.addItem("Zworka")
        self.component.addItem("USB")
        self.component.addItem("Rezonator kwarcowy")
        self.component.addItem("Rezystor")
        self.component.addItem("Cewka")
        self.component.addItem("Złącze")
        self.component.addItem("Tranzystor")
        
        
    def load_pos_file(self, path):
        components = []
        with open(path, 'r') as file:
            for line in file:
                if line.startswith('#') or line.startswith('###') or line.startswith('##'):
                    continue
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 7:
                    ref, val, pkg, x, y, rot, side = parts
                    components.append({
                        "ref": ref,
                        "val": val,
                        "x_mm": float(x),
                        "y_mm": float(y),
                        "rot": float(rot),
                        "side": side
                    })
        return components



    def extract_pcb(self, image, debug=False):
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Nieprawidłowy obraz wejściowy.")

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 🧠 Wybór koloru z GUI
        selected_color = self.pcb_color_select.currentText().lower()

        if "niebieska" in selected_color:
            lower = np.array([90, 40, 40])
            upper = np.array([140, 255, 255])
        elif "zielona" in selected_color:
            lower = np.array([40, 30, 30])
            upper = np.array([85, 255, 255])
        elif "czarna" in selected_color:
            lower = np.array([0, 0, 0])
            upper = np.array([180, 255, 60])
        else:
            raise ValueError(f"Nieznany kolor płytki: {selected_color}")

        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise RuntimeError("❌ Nie wykryto konturu płytki PCB (kolor).")

        pcb_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(pcb_contour)

        print(f"🎯 PCB (kolor={selected_color}) x={x}, y={y}, w={w}, h={h}")

        pcb_img = image[y:y + h, x:x + w]
        rotated = False

        if h > w:
            pcb_img = cv2.rotate(pcb_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rotated = True
            w, h = h, w

        if debug:
            debug_img = image.copy()
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.show_frame(debug_img)
            cv2.imwrite("debug_pcb_mask_color.jpg", pcb_img)

        return pcb_img, rotated, w, h





    def load_model(self, model_path):
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)

            try:
                # Użyj tej samej funkcji co w test.py
                self.model = get_model(2).to(self.device)  # num_classes=2
                self.model.load_state_dict(state_dict, strict=False)
            except RuntimeError as e:
                print("\n❌ Błąd podczas ładowania state_dict:")
                print(e)
                QMessageBox.critical(self, "Błąd", f"Problem z załadowaniem modelu:\n{e}")
                return


            self.model.eval()
            print(f"\nZaładowano model: {model_path}")
        else:
            QMessageBox.critical(self, "Błąd", f"Nie znaleziono modelu: {model_path}")


    def change_model(self, selected_component):
        if selected_component in self.model_paths:
            self.load_model(self.model_paths[selected_component])
            print(f"Załadowano model: {selected_component}")
        else:
            QMessageBox.warning(self, "Uwaga", f"Nie znaleziono modelu dla: {selected_component}")

    def choose_virtual_camera(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        available_cams = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.shape[1] > 100:
                    available_cams.append((i, frame.shape))
                cap.release()

        if not available_cams:
            QMessageBox.warning(self, "Błąd", "Nie znaleziono żadnych dostępnych kamer.")
            return

        cam_options = [f"Kamera {idx} - {shape[1]}x{shape[0]}" for idx, shape in available_cams]
        choice, ok = QInputDialog.getItem(self, "Wybierz kamerę", "Dostępne kamery:", cam_options, 0, False)

        if ok and choice:
            selected_index = int(choice.split()[1])
            self.cap = cv2.VideoCapture(selected_index)
            if self.cap.isOpened():
                self.timer.start(30)
            else:
                QMessageBox.warning(self, "Błąd", f"Nie udało się otworzyć kamery o indeksie {selected_index}.")
        


        if not self.cap.isOpened():
            QMessageBox.critical(self, "Błąd", "Nie można otworzyć wirtualnej kamery.")
            self.cap = None
            return

        self.timer.start(30)

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
        self.analyze = not self.analyze
        self.analyze_button.setText("Wyłącz Analizę" if self.analyze else "Analiza")

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
        tensor_frame = torch.from_numpy(frame / 255.0).permute(2, 0, 1).float().unsqueeze(0).to('cuda')

        with torch.no_grad():
            predictions = self.model(tensor_frame)[0]

        detections = []
        count = 0

        for box, score, label in zip(predictions["boxes"], predictions["scores"], predictions["labels"]):
            if score > 0.7:
                count += 1
                x1, y1, x2, y2 = map(int, box.tolist())
                detections.append({"id": f"ID: {count}", "bbox": (x1, y1, x2, y2), "score": float(score.item())})

        self.count_elements.setText(f"{count}")
        return frame, detections  

    
    def load_pos_file(self, path):
        components = []
        with open(path, 'r') as file:
            for line in file:
                if line.startswith('#') or line.startswith('###') or line.startswith('##'):
                    continue
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 7:
                    ref, val, pkg, x, y, rot, side = parts
                    components.append({
                        "ref": val,
                        "val": val,
                        "x_mm": float(x),
                        "y_mm": float(y),
                        "rot": float(rot),
                        "side": side
                    })
        return components

    def draw_pos_on_pcb(self, pcb_img, components, scale_x, scale_y):
        overlay = pcb_img.copy()
        drawn = 0

        for comp in components:
            x_px = int(comp["x_mm"] * scale_x)
            y_px = int(abs(comp["y_mm"]) * scale_y)  # Y może być ujemne

            if 0 <= x_px < overlay.shape[1] and 0 <= y_px < overlay.shape[0]:
                cv2.circle(overlay, (x_px, y_px), 8, (0, 0, 255), -1)
                cv2.putText(overlay, comp["val"], (x_px + 5, y_px - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                print(f"🎯 {comp['val']} → x={x_px}px, y={y_px}px")
                drawn += 1
            else:
                print(f"⚠️ {comp['val']} poza obrazem!")

        print(f"🖍️ Narysowano komponentów: {drawn}")
        return overlay


    def draw_pos_on_pcb_manual(self, pcb_img, components, scale_x, scale_y, pcb_width_mm, pcb_height_mm):
        # Wylicz granice z .pos
        all_x = [c["x_mm"] for c in components]
        all_y = [abs(c["y_mm"]) for c in components]
        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)
        pos_width = max_x - min_x
        pos_height = max_y - min_y

        # Oblicz automatyczny offset, aby układ był wyśrodkowany na płytce
        offset_board_x = (pcb_width_mm - pos_width) / 2.0
        offset_board_y = (pcb_height_mm - pos_height) / 2.0

        # Ustal dodatkowy, ręczny offset (FINE OFFSET) w mm – możesz modyfikować te stałe lub pobierać z GUI
        FINE_OFFSET_X_MM = -2.5  # Ustaw eksperymentalnie, np. 0.0, 0.5, itp.
        FINE_OFFSET_Y_MM = -3.0

        print(f"🔍 Automatyczny offset: {offset_board_x:.2f} mm (X), {offset_board_y:.2f} mm (Y)")
        print(f"🔍 Fine offset: {FINE_OFFSET_X_MM:.2f} mm (X), {FINE_OFFSET_Y_MM:.2f} mm (Y)")

        overlay = pcb_img.copy()
        drawn = 0

        for comp in components:
            # Przekształć współrzędne:
            # (comp["x_mm"] - min_x) daje pozycję względem minimalnej wartości z pos.
            # Dodajemy offset_board oraz dodatkowy FINE_OFFSET.
            x_mm_adjusted = (comp["x_mm"] - min_x) + offset_board_x + FINE_OFFSET_X_MM
            y_mm_adjusted = (abs(comp["y_mm"]) - min_y) + offset_board_y + FINE_OFFSET_Y_MM

            x_px = int(x_mm_adjusted * scale_x)
            y_px = int(y_mm_adjusted * scale_y)

            # Debug wypisanie
            print(f"🎯 {comp['val']} → x={x_px}px, y={y_px}px (x_mm_adj={x_mm_adjusted:.2f}, y_mm_adj={y_mm_adjusted:.2f})")
            
            if 0 <= x_px < pcb_img.shape[1] and 0 <= y_px < pcb_img.shape[0]:
                cv2.circle(overlay, (x_px, y_px), 4, (0, 0, 255), -1)
                cv2.putText(overlay, comp["val"], (x_px + 3, y_px - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                drawn += 1
            else:
                print(f"⚠️ {comp['val']} poza obrazem!")
      
        print(f"🖍️ Narysowano komponentów: {drawn}")
        return overlay





    def load_and_overlay_pos(self):
        # 1. Wybierz plik .pos
        path, _ = QFileDialog.getOpenFileName(self, "Wybierz plik .pos", "", "Pliki POS (*.pos *.txt)")
        if not path:
            return

        components = self.load_pos_file(path)
        if not components:
            QMessageBox.warning(self, "Błąd", "Nie wczytano żadnych komponentów z pliku .pos.")
            return

        # 2. Pobierz obraz – zamrożony lub z kamery
        if self.frozen_frame is not None:
            frame = self.frozen_frame.copy()
        elif self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                QMessageBox.warning(self, "Błąd", "Nie udało się pobrać obrazu z kamery.")
                return
        else:
            QMessageBox.warning(self, "Błąd", "Brak źródła obrazu.")
            return

        # 3. Wytnij PCB z obrazu
        try:
            pcb_img, rotated, pcb_w_px, pcb_h_px = self.extract_pcb(frame, debug=True)
            print(f"📸 PCB wycięte: {pcb_w_px} x {pcb_h_px} px")
        except Exception as e:
            QMessageBox.warning(self, "Błąd", f"Nie udało się wyciąć PCB:\n{e}")
            return

        # 4. Pobierz fizyczne wymiary płytki od użytkownika
        try:
            pcb_width_mm = float(self.pcb_width_input.text().replace(",", "."))
            pcb_height_mm = float(self.pcb_height_input.text().replace(",", "."))
        except Exception:
            QMessageBox.warning(self, "Błąd", "Nieprawidłowe wymiary płytki PCB.")
            return

        print(f"📏 Wymiary podane: {pcb_width_mm:.2f} mm x {pcb_height_mm:.2f} mm")

        # 5. Oblicz skalę (px/mm) – na podstawie podanych wymiarów
        scale_x = pcb_w_px / pcb_width_mm
        scale_y = pcb_h_px / pcb_height_mm
        print(f"📐 Skala ręczna: {scale_x:.2f} px/mm (X), {scale_y:.2f} px/mm (Y)")

        # 6. Rysuj komponenty na obrazie PCB – overlay z .pos (funkcja draw_pos_on_pcb_manual obsługuje offsety)
        overlayed = self.draw_pos_on_pcb_manual(pcb_img, components, scale_x, scale_y, pcb_width_mm, pcb_height_mm)
        self.current_overlay = overlayed.copy()

        # 7. Uruchom analizę ML na overlayu (obrazie z naniesionymi kropkami z .pos)
        analyzed_img, detections = self.detect_components(overlayed)
        print("🔍 Analiza ML zakończona.")

        # 8. Naniesienie wyników detekcji ML na obrazie overlay – bounding boxy
        result = self.draw_detections_on_overlay(analyzed_img, detections)

        # 9. Zamrożenie i wyświetlenie finalnego obrazu w GUI
        self.frozen_frame = result.copy()
        self.frozen = True
        self.frozen_bboxes = []  # Reset, jeśli potrzebne
        self.show_frame(result)

        # (Debug) Zapisz wynik do pliku
        cv2.imwrite("debug_output_pcb.jpg", result)
        cv2.imwrite("debug_shown_in_gui.jpg", cv2.resize(result, (self.cap_label.width(), self.cap_label.height())))
        print("✅ Zapisano: debug_output_pcb.jpg")





    def draw_detections_on_overlay(self, image, detections):
        overlay = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return overlay
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.critical(self, "Błąd", "Nie udało się pobrać klatki z kamery.")
            self.stop_camera()
            return

        self.frame_count += 1
        original_frame = frame.copy()

        try:
            # 🚨 teraz OK – przekazujemy obraz, nie ścieżkę
            original_frame, rotated, pcb_w_px, pcb_h_px = self.extract_pcb(original_frame, debug=True)
        except Exception as e:
            print(f"⚠️ Błąd przy wycinaniu PCB: {e}")
            original_frame = frame.copy()  # fallback: nieprzetworzony obraz

        if self.analyze:
            frame, detections = self.detect_components(original_frame)
            self.update_component_list(detections)

            self.frozen_frame = frame.copy()
            self.frozen_bboxes = self.bboxes.copy()
            self.frozen = True

        if self.frozen:
            frame = self.frozen_frame.copy()
            bboxes_to_draw = self.frozen_bboxes
        else:
            bboxes_to_draw = self.bboxes

        for bbox in bboxes_to_draw:
            x1, y1, x2, y2 = bbox["bbox"]
            color = bbox["color"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        frame_resized = cv2.resize(frame, (self.cap_label.width(), self.cap_label.height()), interpolation=cv2.INTER_LINEAR)
        h, w, ch = frame_resized.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_resized.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.cap_label.setPixmap(QPixmap.fromImage(qimg))

        if self.recording and self.video_writer:
            self.video_writer.write(frame)



    def show_frame(self, frame):
        frame_resized = cv2.resize(frame, (self.cap_label.width(), self.cap_label.height()), interpolation=cv2.INTER_LINEAR)
        h, w, ch = frame_resized.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_resized.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.cap_label.setPixmap(QPixmap.fromImage(qimg))

        # ✅ Zapisz dokładnie to, co widzi GUI
        cv2.imwrite("debug_shown_in_gui.jpg", frame_resized)




    def show_frozen_frame(self):
        """Wyświetla ostatnią zamrożoną klatkę"""
        if self.frozen_frame is not None:
            self.show_frame(self.frozen_frame)


    def clear_image(self):
        """Resetuje obraz i wznawia działanie kamery"""
        self.frozen = False
        self.frozen_frame = None
        self.cap_label.clear()  # Czyści obrazek
        self.bboxes = []  # Czyści bounding boxy
        self.component_list.clear()  # Czyści listę komponentów




    def update_component_list(self, detections):
        """Aktualizuje listę ID komponentów na podstawie wykrytych obiektów"""
        self.component_list.clear()  # Czyści starą listę
        self.bboxes = []  # Lista boxów

        for i,detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            id_ = f"ID:{i+1}|Score: {detection['score']:.2f}"
            self.component_list.addItem(id_)  # Dodajemy ID do listy 

            # Dodajemy bbox do listy z domyślnym kolorem
            self.bboxes.append({"id": id_, "bbox": (x1, y1, x2, y2), "color": (0, 255, 0), "score": detection["score"]})  # czerwony


    def highlight_bbox(self, item):
        """Zmienia kolor bounding boxa po kliknięciu w ID na liście"""
        clicked_id = item.text()  # Pobieramy ID

        updated = False  # Flaga sprawdzająca, czy znaleziono ID
        for bbox in self.bboxes:
            if bbox["id"] == clicked_id:
                bbox["color"] = (255, 0, 0)  # Zmień kolor na zielony
                updated = True
            if bbox["id"] != clicked_id:
                bbox["color"] = (0, 255, 0)

        if updated:
            self.update_frame()  # Odśwież kamerę
            self.cap_label.repaint()  # Wymuś ponowne narysowanie




    def closeEvent(self, event):
        self.stop_camera()
        event.accept()