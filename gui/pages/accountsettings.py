from PyQt5 import QtWidgets, QtCore
from PyQt5.uic import loadUi
from database import get_user_data, update_email, update_password
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QColor, QPainter, QFont
import re

class AccountSettings(QtWidgets.QWidget):
    def __init__(self, user_id):
        super(AccountSettings, self).__init__()
        loadUi("ui/accountsettings.ui", self)
        self.setAutoFillBackground(True)
        self.user_id = user_id

        # --- Ciemny layout (w formie widgetu) ---
        self.background_widget = QtWidgets.QWidget(self)
        self.background_widget.setGeometry(0, 0, 4000, 4000)  # bardzo duży rozmiar
        self.background_widget.setStyleSheet("background-color: rgb(47, 49, 54);")
        self.background_widget.lower()  # przesuwam na spód
        self.background_widget.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)  # ignoruje kliknięcia
        
        self.initUI()
        self.load_user_data()
        self.saveChanges.clicked.connect(self.save_changes)
        
        # Initialize status message
        self.statusLabel.setText("")
        
        # Connect field changes to clear status
        self.new_email.textChanged.connect(lambda: self.statusLabel.setText(""))
        self.password.textChanged.connect(lambda: self.statusLabel.setText(""))
        self.new_password.textChanged.connect(lambda: self.statusLabel.setText(""))
        self.confirm_new_password.textChanged.connect(lambda: self.statusLabel.setText(""))

    def initUI(self):
        # Set tab order for better navigation
        self.setTabOrder(self.new_email, self.password)
        self.setTabOrder(self.password, self.new_password)
        self.setTabOrder(self.new_password, self.confirm_new_password)
        self.setTabOrder(self.confirm_new_password, self.saveChanges)
        
        # Add password validation requirements as tooltips
        self.new_password.setToolTip("Password must be at least 8 characters long and include uppercase, lowercase, digit, and special character")
        
        # Placeholder initialization for avatar - will be updated with user data
        self.avatarLabel.setText("")

    def set_avatar_initials(self, name, surname):
        """Set the avatar label with user's initials"""
        if not name or not surname:
            self.avatarLabel.setText("?")
            return
            
        initials = (name[0] + surname[0]).upper()
        self.avatarLabel.setText(initials)
    
    def set_status(self, message, status_type="info"):
        """Set status message with type (info, warning, error)"""
        self.statusLabel.setText(message)
        
        if status_type == "warning":
            self.statusLabel.setProperty("warning", True)
            self.statusLabel.setProperty("error", False)
        elif status_type == "error":
            self.statusLabel.setProperty("warning", False)
            self.statusLabel.setProperty("error", True)
        else:  # info
            self.statusLabel.setProperty("warning", False)
            self.statusLabel.setProperty("error", False)
        
        # Force style update
        self.statusLabel.style().unpolish(self.statusLabel)
        self.statusLabel.style().polish(self.statusLabel)

    def load_user_data(self):
        user_data = get_user_data(self.user_id)
        if user_data:
            name = user_data.get('name', '')
            surname = user_data.get('surname', '')
            position = str(user_data.get('position_name', ''))
            email = user_data.get('email', '')
            
            # Set user data in UI
            self.name.setText(name)
            self.surname.setText(surname)
            self.position.setText(position)
            self.email.setText(email)
            
            # Set profile header information
            self.usernameLabel.setText(f"{name} {surname}")
            self.emailLabel.setText(email)
            self.positionLabel.setText(position)
            
            # Set avatar initials
            self.set_avatar_initials(name, surname)
        else:
            self.set_status("Failed to load user data", "error")
            QMessageBox.critical(self, "Error", "Failed to load user data")

    def is_valid_email(self, email):
        """Validate email format"""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return re.match(pattern, email) is not None
    
    def is_valid_password(self, password):
        """Validate password requirements"""
        # At least 8 characters long with uppercase, lowercase, digit, and special char
        pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&.,])[A-Za-z\d@$!%*?&.,]{8,}$"
        return re.match(pattern, password) is not None

    def save_changes(self):
        new_email = self.new_email.text().strip()
        current_password = self.password.text().strip()
        new_password = self.new_password.text().strip()
        confirm_new_password = self.confirm_new_password.text().strip()
        
        # Clear status initially
        self.statusLabel.setText("")
        has_changes = False
        
        # Email change
        if new_email:
            if not self.is_valid_email(new_email):
                self.set_status("Invalid email format", "error")
                return
                
            if new_email != self.email.text().strip():
                has_changes = True
                if update_email(self.user_id, new_email):
                    # Update UI with new email
                    self.email.setText(new_email)
                    self.emailLabel.setText(new_email)
                    self.set_status("Email updated successfully")
                    self.new_email.clear()
                else:
                    self.set_status("Failed to update email", "error")
                    return
        
        # Password change
        if new_password or confirm_new_password or current_password:
            # Check if all fields are provided
            if not (new_password and confirm_new_password and current_password):
                self.set_status("All password fields are required for password change", "warning")
                return
                
            # Password validation
            if new_password != confirm_new_password:
                self.set_status("New passwords do not match", "error")
                return
                
            if not self.is_valid_password(new_password):
                self.set_status("Password must meet security requirements", "error")
                return
                
            has_changes = True
            if update_password(self.user_id, current_password, new_password):
                self.set_status("Password updated successfully")
                # Clear password fields
                self.password.clear()
                self.new_password.clear()
                self.confirm_new_password.clear()
            else:
                self.set_status("Failed to update password. Check your current password.", "error")
                return
        
        # If no changes were attempted
        if not has_changes:
            self.set_status("No changes were made", "warning")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = AccountSettings(user_id=1)
    window.show()
    sys.exit(app.exec_())