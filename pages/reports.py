from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi

class Reports(QDialog):
    def __init__(self):
        super(Reports, self).__init__()
        loadUi("ui/report.ui", self)
