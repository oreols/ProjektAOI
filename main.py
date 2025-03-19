import sys
from PyQt5.QtWidgets import QApplication, QStackedWidget
from pages.login import Login

app = QApplication(sys.argv)
widget = QStackedWidget()

loginPage = Login(widget)
widget.addWidget(loginPage)

widget.setFixedWidth(1200)
widget.setFixedHeight(800)

widget.show()
sys.exit(app.exec_())
