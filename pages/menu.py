from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi
from pages.reports import Reports
from pages.camera import Camera
from pages.history import History
from pages.account import Account
from pages.register import Register
from pages.accountsettings import AccountSettings


class MainWindow(QMainWindow):
    def __init__(self, role, widget, user_id):
        super(MainWindow, self).__init__()
        loadUi("ui/menu.ui", self)
        self.role = role
        self.user_id = user_id
        self.widget = widget

        # Dodanie widżetów do stackedWidget
        self.stackedWidget.addWidget(Reports())  # Index 0
        self.stackedWidget.addWidget(Camera())  # Index 1
        self.stackedWidget.addWidget(History())  # Index 2
        self.stackedWidget.addWidget(Account())  # Index 3
        self.stackedWidget.addWidget(AccountSettings(self.user_id))  # Index 4
        if self.role == "admin":
            self.stackedWidget.addWidget(Register())  # Index 5 (tylko dla admina)

        # Połączenia akcji z widżetami
        self.actionReports.triggered.connect(lambda: self.stackedWidget.setCurrentIndex(2))
        self.actionCameras.triggered.connect(lambda: self.stackedWidget.setCurrentIndex(3))
        self.actionHistory.triggered.connect(lambda: self.stackedWidget.setCurrentIndex(4))
        self.actionAccount.triggered.connect(lambda: self.stackedWidget.setCurrentIndex(5))
        self.actionRegister.triggered.connect(lambda: self.stackedWidget.setCurrentIndex(7))
        self.actionAccountSettings.triggered.connect(lambda: self.stackedWidget.setCurrentIndex(6))

        # Admin widzi więcej opcji
        if self.role == "admin":
            self.actionRegister.triggered.connect(lambda: self.stackedWidget.setCurrentIndex(7))
            self.actionAccount.setVisible(True)
            self.actionRegister.setVisible(True)
            self.actionAccountSettings.setVisible(True)
        elif self.role == "user":
            self.actionAccount.setVisible(False)
            self.actionRegister.setVisible(False)
            self.actionAccountSettings.setVisible(True)

        # Akcja wylogowania
        self.actionLogout.triggered.connect(self.logout)

    def logout(self):
        print("Wylogowano!")
        self.widget.setCurrentIndex(0)
