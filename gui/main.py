import sys
from PyQt5.QtWidgets import QApplication, QStackedWidget, QSystemTrayIcon
from PyQt5.QtGui import QIcon
from pages.login import Login
from pages.camera import Camera

# Ustawienia dla polskich znaków
import os
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(os.path.dirname(sys.executable), "plugins", "platforms")
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

app = QApplication(sys.argv)
app.setApplicationName("System Analizy PCB - AOI")

# Utworzenie katalogu assets/icons jeśli nie istnieje
icons_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icons")
os.makedirs(icons_dir, exist_ok=True)

# Ustawienie ikony aplikacji
app.tray_icon = QSystemTrayIcon(QIcon("assets/icons/logo.png"), app)
app.tray_icon.show()
app.setWindowIcon(QIcon("assets/icons/logo.png"))

widget = QStackedWidget()
widget.setWindowTitle("System Analizy PCB - AOI")

loginPage = Login(widget)
widget.addWidget(loginPage)

widget.setFixedWidth(1400)
widget.setFixedHeight(900)

widget.show()
sys.exit(app.exec_())
