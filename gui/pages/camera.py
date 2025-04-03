from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
import cv2
import os
from PyQt5.QtWidgets import QDialog, QLabel, QMessageBox, QFileDialog, QListWidget
import torchvision
import torch
from PyQt5.QtCore import QTimer
from models.faster_rcnn import get_model
import urllib.request

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
        self.component.addItem("Kondensator")
        self.component.addItem("Układ scalony")
        self.component.addItem("Zworka")
        self.component.addItem("USB")
        self.component.addItem("Rezonator kwarcowy")
        self.component.addItem("Rezystor")
        self.component.addItem("Cewka")
        self.component.addItem("Złącze")
        self.component.addItem("Tranzystor")




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

        
        # for i in range(5):  # Testujemy kamery od 0 do 4
        #     cap = cv2.VideoCapture(i)
        #     if cap.isOpened():
        #         print(f"Znaleziono kamerę: {i}")
        #         cap.release()  # Zamykamy kamerę po sprawdzeniu
        #     else:
        #         print(f"Brak kamery na indeksie {i}")

        virtual_camera_index = 2
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
        tensor_frame = torch.from_numpy(frame / 255.0).permute(2, 0, 1).float().unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(tensor_frame)[0]

        detections = []
        count = 0

        for box, score, label in zip(predictions["boxes"], predictions["scores"], predictions["labels"]):
            if score > 0.8:
                count += 1
                x1, y1, x2, y2 = map(int, box.tolist())
                detections.append({"id": f"ID: {count}", "bbox": (x1, y1, x2, y2), "score": float(score.item())})

        self.count_elements.setText(f"{count}")
        return frame, detections  


    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.critical(self, "Błąd", "Nie udało się pobrać klatki z kamery.")
            self.stop_camera()
            return

        self.frame_count += 1
        original_frame = frame.copy()

        if self.analyze:
            frame, detections = self.detect_components(original_frame)  # Pobieramy detekcje
            self.update_component_list(detections)  # Aktualizujemy listę ID komponentów

            # **Zamrażamy klatkę po analizie**  
            self.frozen_frame = frame.copy()  
            self.frozen_bboxes = self.bboxes.copy()  
            self.frozen = True  # Ustawiamy zamrożenie

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
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Wyświetlanie obrazu
        frame_resized = cv2.resize(frame, (self.cap_label.width(), self.cap_label.height()), interpolation=cv2.INTER_LINEAR)
        h, w, ch = frame_resized.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_resized.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.cap_label.setPixmap(QPixmap.fromImage(qimg))

        if self.recording and self.video_writer:
            self.video_writer.write(frame)  # Zapisujemy wideo



    def show_frame(self, frame):
        """Funkcja pomocnicza do wyświetlania obrazu"""
        frame_resized = cv2.resize(frame, (self.cap_label.width(), self.cap_label.height()), interpolation=cv2.INTER_LINEAR)
        h, w, ch = frame_resized.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_resized.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.cap_label.setPixmap(QPixmap.fromImage(qimg))

        if self.recording and self.video_writer:
            self.video_writer.write(frame) 


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