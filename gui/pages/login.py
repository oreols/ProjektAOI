from PyQt5.QtWidgets import QDialog, QLineEdit
from PyQt5.uic import loadUi
from database import verify_login
from pages.menu import MainWindow

class Login(QDialog):
    def __init__(self, widget):
        super(Login, self).__init__()
        loadUi("ui/login.ui", self)
        self.widget = widget
        self.loginButton.clicked.connect(self.loginFunction)
        self.password.setEchoMode(QLineEdit.Password)

    def loginFunction(self):
        email = self.email.text().strip()
        password = self.password.text().strip()

        if not email or not password:
            self.error.setText("Uzupełnij wszystkie pola")
            return

        role, user_id = verify_login(email, password)

        if user_id is None:
            self.error.setText("Niepoprawne dane logowania")
            return

        print("Login Success, id:", user_id, "role:", role)
        mainWindow = MainWindow("admin" if role == 1 else "user", self.widget, user_id)
        self.widget.addWidget(mainWindow)
        self.widget.setCurrentWidget(mainWindow)


