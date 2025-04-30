import sys
from PyQt5.QtWidgets import QApplication, QStackedWidget
from pages.login import Login
from pages.camera import Camera

app = QApplication(sys.argv)
widget = QStackedWidget()

loginPage = Login(widget)
widget.addWidget(loginPage)

widget.setFixedWidth(1400)
widget.setFixedHeight(900)

widget.show()
sys.exit(app.exec_())