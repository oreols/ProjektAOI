from PyQt5.QtWidgets import QDialog, QLineEdit
from PyQt5.uic import loadUi
from database import create_user, get_positions
import bcrypt
import re
pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&.,])[A-Za-z\d@$!%*?&.,]{8,}$"

class Register(QDialog):
    def __init__(self):
        super(Register, self).__init__()
        loadUi("ui/register.ui", self)
        self.setup_ui()
        self.addAccountButton.clicked.connect(self.registerFunction)
        self.password.setEchoMode(QLineEdit.Password)
        self.confirm_password.setEchoMode(QLineEdit.Password)

    def setup_ui(self):
        self.permisions.setChecked(False)
        positions = get_positions()
        
        for pos in positions:
            self.position.addItem(pos['name'], pos['id'])

    def registerFunction(self):
        email = self.email.text().strip()
        
        password = self.password.text().strip()
        confirm_password = self.confirm_password.text().strip()
        
        position_id = self.position.currentData()
        permissions = 1 if self.permisions.isChecked() else 0 
        name = self.name.text().strip()
        surname = self.surname.text().strip()

        # Walidacja danych wejściowych
        if not email or not name or not surname or not password or not confirm_password or not position_id:
            self.error_msg.setText("Uzupełnij wszystkie pola.")
            return
        
        if name.isdigit() or surname.isdigit():
            self.error_msg.setText("Imię i nazwisko nie mogą zawierać cyfr.")
            return
        
        if len(name) < 3 or len(surname) < 3:
            self.error_msg.setText("Imię i nazwisko muszą zawierać co najmniej 3 znaki.")
            return
        
        if not re.match(pattern, password):
            self.error_msg.setText("Hasło nie spełnia wymagań.")
            return

        if password != confirm_password:
            self.error_msg.setText("Hasła nie są identyczne.")
            return
        
        if "@" not in email:
            self.error_msg.setText("Niepoprawny adres email.")
            return

        # Szyfrowanie hasła
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Dodanie użytkownika do bazy
        success = create_user(email, hashed_password, position_id, permissions, name, surname)
        if success:
            self.error_msg.setText("Konto zostało utworzone.")
            self.email.clear()
            self.password.clear()
            self.confirm_password.clear()
            self.position.clear()
            self.permisions.setChecked(False)
            self.name.clear()
            self.surname.clear()
            self.setup_ui()

        else:
            self.error_msg.setText("Nie udało się utworzyć konta.")
