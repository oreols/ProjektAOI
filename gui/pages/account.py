from PyQt5.QtWidgets import QDialog, QTableWidgetItem, QPushButton, QMessageBox
from PyQt5.uic import loadUi
from database import get_users,delete_user

class Account(QDialog):
    def __init__(self):
        super(Account, self).__init__()
        loadUi("ui/Account.ui", self)
        self.setup_ui()

    def setup_ui(self):
        users = get_users()
        self.tableWidget.setRowCount(0)

        for row_num, user in enumerate(users):
            self.tableWidget.insertRow(row_num)
            self.tableWidget.setItem(row_num, 0, QTableWidgetItem(user['name']))
            self.tableWidget.setItem(row_num, 1, QTableWidgetItem(user['surname']))
            self.tableWidget.setItem(row_num, 2, QTableWidgetItem(user['email']))
            self.tableWidget.setItem(row_num, 3, QTableWidgetItem((user['position_name'])))
            role_text = "admin" if user['admin'] == 1 else "użytkownik"
            self.tableWidget.setItem(row_num, 4, QTableWidgetItem(role_text))

            # Przycisk Edytuj
            edit_button = QPushButton("Edytuj")
            edit_button.clicked.connect(lambda checked, u=user: self.edit_user(u['id']))
            self.tableWidget.setCellWidget(row_num, 5, edit_button)

            # Przycisk Usuń
            delete_button = QPushButton("Usuń")
            delete_button.clicked.connect(lambda checked, u=user: self.del_user(u['id']))
            self.tableWidget.setCellWidget(row_num, 6, delete_button)

    def edit_user(self, user_id):
        QMessageBox.information(self, "Edytuj Użytkownika", f"Edytowanie użytkownika ID: {user_id}")
        # Tutaj dodaj logikę edycji

    def del_user(self, user_id):
        reply = QMessageBox.question(self, 'Usuń Użytkownika', f'Czy na pewno chcesz usunąć użytkownika ID: {user_id}?',
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if delete_user(user_id):
                QMessageBox.information(self, "Usuń Użytkownika", "Użytkownik został usunięty.")
                self.setup_ui()
            else:
                QMessageBox.warning(self, "Usuń Użytkownika", "Błąd podczas usuwania użytkownika.")

    
