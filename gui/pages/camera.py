from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
import sys
import cv2
import os
from PyQt5.QtWidgets import QDialog, QLabel, QApplication, QMessageBox, QFileDialog
import torchvision
import torch
from PyQt5.QtCore import QTimer

class Camera(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("ui/Camera.ui", self)
        self.setGeometry(100, 100, 1200, 800)
        self.cap_label = self.findChild(QLabel, "cap")

        self.model_paths = {
            "Kondensator": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/trained_components/final_capacitor_faster_rcnn_pcb.pth")),
            "Układ scalony": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/trained_components/ic_faster_rcnn_pcb.pth")),
            "Zworka": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/trained_components/jumpers_faster_rcnn_pcb.pth")),
            "USB": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/models/trained_components/usb_faster_rcnn_pcb.pth")),
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
        self.component.addItem("Kondensator")
        self.component.addItem("Układ scalony")
        self.component.addItem("Zworka")
        self.component.addItem("USB")

    def load_model(self, model_path):
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                self.model.eval()
                print(f"Załadowano model: {model_path}")
            else:
                QMessageBox.critical(self, "Błąd", f"Nie znaleziono modelu: {model_path}")

        # Podpięcie wyboru komponentu do zmiany modelu:
            self.component.currentTextChanged.connect(self.change_model)

    def change_model(self, selected_component):
        if selected_component in self.model_paths:
            self.load_model(self.model_paths[selected_component])
            print(f"Załadowano model: {selected_component}")
        else:
            QMessageBox.warning(self, "Uwaga", f"Nie znaleziono modelu dla: {selected_component}")

    def choose_virtual_camera(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        virtual_camera_index = 1 
        self.cap = cv2.VideoCapture(virtual_camera_index)

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

        self.cap = cv2.VideoCapture(0)
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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_frame = torch.from_numpy(rgb_frame / 255.0).permute(2, 0, 1).float().unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(tensor_frame)[0]

        count = 0
        for box, score, label in zip(predictions["boxes"], predictions["scores"], predictions["labels"]):
            if score > 0.8:
                count += 1
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        self.count_elements.setText(f"{count}")
        return frame

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.critical(self, "Błąd", "Nie udało się pobrać klatki z kamery.")
            self.stop_camera()
            return

        self.frame_count += 1
        original_frame = frame.copy()

        if self.analyze:
            frame = self.detect_components(original_frame)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Rozciąganie obrazu do pełnych wymiarów labela cap
        frame = cv2.resize(frame, (self.cap_label.width(), self.cap_label.height()), interpolation=cv2.INTER_LINEAR)

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.cap_label.setPixmap(QPixmap.fromImage(qimg))

        if self.recording and self.video_writer:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(bgr_frame)

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()