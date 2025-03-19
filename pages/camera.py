from PyQt5.QtWidgets import QDialog
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QTransform
from PyQt5.uic import loadUi
import os

class Camera(QDialog):
    def __init__(self):
        super(Camera, self).__init__()
        loadUi("ui/camera.ui", self)
        self.image_folder = "./images"
        self.image_list = [f for f in os.listdir(self.image_folder) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
        self.current_index = 0
        self.rotationAngle = 270
        self.scale_factor = 0.4
        self.offset = QPoint(0, 0)
        self.drag_start = None

        self.loadImage()
        self.rotationButton.clicked.connect(self.rotateImage)
        self.nextButton.clicked.connect(self.nextImage)
        self.previousButton.clicked.connect(self.previousImage)

    def loadImage(self):
        if not self.image_list:
            return
        self.currentFile = os.path.join(self.image_folder, self.image_list[self.current_index])
        pixmap = QPixmap(self.currentFile)
        if not pixmap.isNull():
            self.pixmap = pixmap
            self.updateImage()
            self.imgName.setText(os.path.basename(self.currentFile))

    def updateImage(self):
        if hasattr(self, 'pixmap'):
            transformed_pixmap = self.pixmap.transformed(QTransform().rotate(self.rotationAngle))
            scaled_pixmap = transformed_pixmap.scaled(self.pixmap.size() * self.scale_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.imgSrc.setPixmap(scaled_pixmap)

    def rotateImage(self):
        self.rotationAngle += 90
        if self.rotationAngle >= 360:
            self.rotationAngle = 0
        self.updateImage()

    def nextImage(self):
        self.current_index = (self.current_index + 1) % len(self.image_list)
        self.rotationAngle = -90
        self.scale_factor = 0.4
        self.offset = QPoint(0, 0)
        self.loadImage()

    def previousImage(self):
        self.current_index = (self.current_index - 1) % len(self.image_list)
        self.rotationAngle = -90
        self.scale_factor = 0.4
        self.offset = QPoint(0, 0)
        self.loadImage()
