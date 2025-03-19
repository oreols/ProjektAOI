from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QVBoxLayout, QTableView, QHeaderView, QLineEdit
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt, QSortFilterProxyModel

class History(QDialog):
    def __init__(self):
        super(History, self).__init__()
        loadUi("ui/history.ui", self)

        mainLayout = QVBoxLayout()
        numberPCB = ('PCB-20240313-8471','SN-9832-5567-PCB','PCB-XL2025-3746','5A9B-PCB-6243','PCB-2025-AB4732','SN-PCB-920374','PCB-5581-CX2024','3F72-PCB-1049','PCB-YY812-7654','SN-4573-PCB-2025')
        model = QStandardItemModel(len(numberPCB), 1)
        model.setHorizontalHeaderLabels(['Numer PCB'])
        for row, number in enumerate(numberPCB):
            item = QStandardItem(number)
            model.setItem(row, 0, item)

        filter_proxy_model = QSortFilterProxyModel()
        filter_proxy_model.setSourceModel(model)
        filter_proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        filter_proxy_model.setFilterKeyColumn(0)

        search_field = QLineEdit()
        search_field.setStyleSheet("font-size: 20px; height: 60px;")
        mainLayout.addWidget(search_field)

        table = QTableView()
        table.setStyleSheet("font-size: 20px;")
        search_field.textChanged.connect(filter_proxy_model.setFilterFixedString)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setModel(filter_proxy_model)
        mainLayout.addWidget(table)

        self.setLayout(mainLayout)