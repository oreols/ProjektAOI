from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from database import get_user_data, update_email, update_password
from PyQt5.QtWidgets import QMessageBox

class AccountSettings(QtWidgets.QWidget):
    def __init__(self, user_id):
        super(AccountSettings, self).__init__()
        loadUi("ui/accountsettings.ui", self)
        self.user_id = user_id
        self.load_user_data()
        self.saveChanges.clicked.connect(self.save_changes)

    def load_user_data(self):
        user_data = get_user_data(self.user_id)
        if user_data:
            self.name.setText(user_data.get('name', ''))
            self.surname.setText(user_data.get('surname', ''))
            self.position.setText(str(user_data.get('position_name', '')))
            self.email.setText(user_data.get('email', ''))
        else:
            QMessageBox.critical(self, "Błąd", "Nie udało się załadować danych użytkownika.")

    def save_changes(self):
        new_email = self.new_email.text().strip()
        current_password = self.password.text().strip()
        new_password = self.new_password.text().strip()
        confirm_new_password = self.confirm_new_password.text().strip()
        

        if new_email:
            if new_email != self.email.text().strip():
                if update_email(self.user_id, new_email):
                    QMessageBox.information(self, "Sukces", "Email został zmieniony.")
                    self.email.setText(new_email)
                else:
                    QMessageBox.critical(self, "Błąd", "Nie udało się zmienić emailu.")

        if current_password and new_password and confirm_new_password:
            if new_password != confirm_new_password:
                QMessageBox.critical(self, "Błąd", "Nowe hasła nie są identyczne.")
                return
            if update_password(self.user_id, current_password, new_password):
                QMessageBox.information(self, "Sukces", "Hasło zostało zmienione.")
            else:
                QMessageBox.critical(self, "Błąd", "Nie udało się zmienić hasła. Sprawdź obecne hasło.")

        self.new_email.clear()
        self.password.clear()
        self.new_password.clear()
        self.confirm_new_password.clear()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = AccountSettings(user_id=1)
    window.show()
    sys.exit(app.exec_())