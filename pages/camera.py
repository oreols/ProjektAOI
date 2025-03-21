from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
import sys
import cv2
from PyQt5.QtWidgets import QDialog,QLabel, QApplication, QMessageBox, QFileDialog
from PyQt5.QtCore import QTimer

class Camera(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("ui/Camera.ui", self)
        self.setGeometry(100, 100, 1200, 800)
        self.cap_label = self.findChild(QLabel, "cap")


        # Camera
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.analyze = False
        self.recording = False
        self.video_writer = None

        # Button actions
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.analyze_button.clicked.connect(self.toggle_analysis)
        self.record_button.clicked.connect(self.toggle_recording)
        self.virtual_cam_button.clicked.connect(self.choose_virtual_camera)

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
        self.cap_label.setPixmap(QPixmap())  # Wyczyść podgląd kamery
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
        else:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Zapisz nagranie", "", "AVI Files (*.avi);;All Files (*)", options=options)
            if file_name:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(file_name, fourcc, 30.0, (640, 480))
                self.recording = True
                self.record_button.setText("Zatrzymaj Nagrywanie")
    def choose_virtual_camera(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Wybierz plik wideo", "", "Video Files (*.mp4 *.avi *.mov)", options=options)
        
        if file_name:
            self.cap = cv2.VideoCapture(file_name)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Błąd", "Nie można otworzyć pliku wideo.")
                self.cap = None
                return
            self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.critical(self, "Błąd", "Nie udało się pobrać klatki z kamery.")
            self.stop_camera()
            return

        # Convert frame to RGB and display it
        if self.analyze:
            frame = cv2.Canny(frame, 100, 200)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.cap_label.setPixmap(QPixmap.fromImage(qimg))
  # Display frame in QLabel

        # Save frame if recording
        if self.recording and self.video_writer:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(bgr_frame)

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


