from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi

class Account(QDialog):
    def __init__(self):
        super(Account, self).__init__()
        loadUi("ui/Account.ui", self)
