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

        # Dodanie widżetów do stackedWidget z zapisaniem ich indeksów
        self.index_reports = self.stackedWidget.addWidget(Reports(user_id=self.user_id))
        self.index_camera = self.stackedWidget.addWidget(Camera())
        self.index_history = self.stackedWidget.addWidget(History())
        self.index_account = self.stackedWidget.addWidget(Account())
        self.index_account_settings = self.stackedWidget.addWidget(AccountSettings(self.user_id))

        self.index_register = None
        if self.role == "admin":
            self.index_register = self.stackedWidget.addWidget(Register())

        # Połączenia akcji z widżetami (odpowiednie indeksy)
        self.actionReports.triggered.connect(lambda: self.stackedWidget.setCurrentIndex(self.index_reports))
        self.actionCameras.triggered.connect(lambda: self.stackedWidget.setCurrentIndex(self.index_camera))
        self.actionHistory.triggered.connect(lambda: self.stackedWidget.setCurrentIndex(self.index_history))
        self.actionAccount.triggered.connect(lambda: self.stackedWidget.setCurrentIndex(self.index_account))
        self.actionAccountSettings.triggered.connect(lambda: self.stackedWidget.setCurrentIndex(self.index_account_settings))

        if self.role == "admin" and self.index_register is not None:
            self.actionRegister.triggered.connect(lambda: self.stackedWidget.setCurrentIndex(self.index_register))

        # Widoczność akcji w zależności od roli
        self.actionAccount.setVisible(self.role == "admin")
        self.actionRegister.setVisible(self.role == "admin")
        self.actionAccountSettings.setVisible(True)

        # Akcja wylogowania
        self.actionLogout.triggered.connect(self.logout)

        # Automatyczne przełączenie na widok kamery po zalogowaniu
        self.stackedWidget.setCurrentIndex(self.index_camera)

    def logout(self):
        print("Wylogowano!")
        self.widget.setCurrentIndex(0)