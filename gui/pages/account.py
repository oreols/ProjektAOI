from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QMessageBox, QFrame, QSpacerItem, QSizePolicy, QWidget)
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi
from database import get_users, delete_user # Upewnij się, że to jest poprawny import

class UserCard(QFrame):
    def __init__(self, user_data, parent_dialog):
        super().__init__()
        self.user_data = user_data
        self.parent_dialog = parent_dialog # Referencja do głównego dialogu (Account)

        self.setProperty("class", "UserCard") # Do stylizacji przez QSS: QFrame.UserCard

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10) # Wewnętrzne marginesy karty
        main_layout.setSpacing(8)

        # Imię i Nazwisko
        name_text = f"{user_data.get('name', 'N/A')} {user_data.get('surname', 'N/A')}"
        self.name_label = QLabel(name_text)
        self.name_label.setObjectName(f"userNameLabel_{user_data.get('id')}") # Unikalna nazwa obiektu
        self.name_label.setWordWrap(True)
        main_layout.addWidget(self.name_label)

        # Email
        self.email_label = QLabel(f"Email: {user_data.get('email', 'N/A')}")
        self.email_label.setObjectName(f"userEmailLabel_{user_data.get('id')}")
        self.email_label.setWordWrap(True)
        main_layout.addWidget(self.email_label)

        # Stanowisko
        self.position_label = QLabel(f"Stanowisko: {user_data.get('position_name', 'N/A')}")
        self.position_label.setObjectName(f"userPositionLabel_{user_data.get('id')}")
        self.position_label.setWordWrap(True)
        main_layout.addWidget(self.position_label)
        
        # Rola
        role_text = "Administrator" if user_data.get('admin') == 1 else "Użytkownik"
        self.role_label = QLabel(f"Rola: {role_text}")
        self.role_label.setObjectName(f"userRoleLabel_{user_data.get('id')}")
        main_layout.addWidget(self.role_label)

        # Kontener na przyciski
        button_container = QWidget()
        buttons_layout = QHBoxLayout(button_container)
        buttons_layout.setContentsMargins(0, 8, 0, 0) # Odstęp nad przyciskami
        buttons_layout.setSpacing(10)
        buttons_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))


        # Przycisk Edytuj
        self.edit_button = QPushButton("Edytuj")
        self.edit_button.setCursor(Qt.PointingHandCursor)
        self.edit_button.clicked.connect(self.handle_edit)
        buttons_layout.addWidget(self.edit_button)

        # Przycisk Usuń
        self.delete_button = QPushButton("Usuń")
        self.delete_button.setCursor(Qt.PointingHandCursor)
        self.delete_button.clicked.connect(self.handle_delete)
        buttons_layout.addWidget(self.delete_button)
        
        main_layout.addWidget(button_container)

    def handle_edit(self):
        self.parent_dialog.edit_user(self.user_data['id'])

    def handle_delete(self):
        self.parent_dialog.del_user(self.user_data['id'])


class Account(QDialog):
    def __init__(self):
        super(Account, self).__init__()
        loadUi("ui/Account.ui", self) # Upewnij się, że ścieżka jest poprawna
        self.load_and_display_users()

    def clear_layout(self, layout):
        """Usuwa wszystkie widgety z danego layoutu."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else: # Jeśli to spacer
                    layout.removeItem(item)


    def load_and_display_users(self):
        # Wyczyść poprzednie karty z userListLayout
        # self.userListLayout to nazwa QVBoxLayout wewnątrz scrollAreaWidgetContents
        self.clear_layout(self.userListLayout)

        users = get_users()
        if not users:
            # Można dodać QLabel z informacją "Brak użytkowników"
            info_label = QLabel("Brak zdefiniowanych użytkowników.")
            info_label.setAlignment(Qt.AlignCenter)
            self.userListLayout.addWidget(info_label) # Dodajemy info
            if self.userListLayout.count() > 1: # jeśli jest info i spacer
                 item = self.userListLayout.takeAt(self.userListLayout.count() -1) # Usuń stary spacer
                 if item is not None:
                     self.userListLayout.removeItem(item)

        for user_data in users:
            card = UserCard(user_data, self)
            # Wstawiamy kartę PRZED ostatnim elementem (którym jest spacer)
            self.userListLayout.insertWidget(self.userListLayout.count() - 1, card)

        if self.userListLayout.count() == 0 or not isinstance(self.userListLayout.itemAt(self.userListLayout.count() -1).widget(), QSpacerItem):
             if users: # Dodaj spacer tylko jeśli są jacyś użytkownicy
                spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
                self.userListLayout.addSpacerItem(spacer)


    def edit_user(self, user_id):
        # Tutaj Twoja logika edycji, np. otwarcie nowego dialogu z danymi użytkownika
        QMessageBox.information(self, "Edytuj Użytkownika", f"Rozpoczynanie edycji użytkownika ID: {user_id}")
        # Po edycji, odśwież listę:
        # self.load_and_display_users() # Jeśli edycja odbywa się w innym oknie

    def del_user(self, user_id):
        reply = QMessageBox.question(self, 'Usuń Użytkownika', 
                                     f'Czy na pewno chcesz usunąć użytkownika ID: {user_id}?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if delete_user(user_id): # Twoja funkcja usuwająca z bazy
                QMessageBox.information(self, "Usuń Użytkownika", "Użytkownik został usunięty.")
                self.load_and_display_users() # Odśwież widok
            else:
                QMessageBox.warning(self, "Usuń Użytkownika", "Błąd podczas usuwania użytkownika.")

# Jeśli uruchamiasz ten plik bezpośrednio do testów:
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    # test
    _mock_users_db = [
        {'id': 1, 'name': 'Jan', 'surname': 'Kowalski', 'email': 'jan.kowalski@example.com', 'position_name': 'Programista', 'admin': 0},
        {'id': 2, 'name': 'Anna', 'surname': 'Nowak', 'email': 'anna.nowak@example.com', 'position_name': 'Tester', 'admin': 1},
        {'id': 3, 'name': 'Piotr', 'surname': 'Zieliński', 'email': 'piotr.zielinski@example.com', 'position_name': 'Projektant UX/UI Designer Developer Extraordinaire', 'admin': 0},
    ]
    _next_id = 4

    def get_users():
        return list(_mock_users_db)

    def delete_user(user_id):
        global _mock_users_db
        original_len = len(_mock_users_db)
        _mock_users_db = [u for u in _mock_users_db if u['id'] != user_id]
        return len(_mock_users_db) < original_len


    app = QApplication(sys.argv)
    dialog = Account()
    dialog.show()
    sys.exit(app.exec_())