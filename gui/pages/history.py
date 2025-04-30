from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt, QSortFilterProxyModel

class History(QDialog):
    def __init__(self):
        super(History, self).__init__()
        
        # Wczytanie UI z pliku
        loadUi("ui/history.ui", self)
        
        # Pobieranie danych z bazy (symulowane)
        # Tutaj można zastąpić logiką pobierania z bazy danych
        numberPCB = ('PCB-20240313-8471','SN-9832-5567-PCB','PCB-XL2025-3746',
                     '5A9B-PCB-6243','PCB-2025-AB4732','SN-PCB-920374',
                     'PCB-5581-CX2024','3F72-PCB-1049','PCB-YY812-7654','SN-4573-PCB-2025')
        
        # Przygotowanie modelu danych
        model = QStandardItemModel(len(numberPCB), 1)
        model.setHorizontalHeaderLabels(['Numer PCB'])
        for row, number in enumerate(numberPCB):
            item = QStandardItem(number)
            model.setItem(row, 0, item)
            
        # Przygotowanie modelu do filtrowania
        filter_proxy_model = QSortFilterProxyModel()
        filter_proxy_model.setSourceModel(model)
        filter_proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        filter_proxy_model.setFilterKeyColumn(0)
        
        # Ustawienie modelu dla istniejącej tabeli w UI
        self.tableWidget.setModel(filter_proxy_model)
        
        # Podłączenie pola wyszukiwania do filtra
        self.search_field.textChanged.connect(filter_proxy_model.setFilterFixedString)
        
        # Podłączenie przycisku powrotu
        self.backButton.clicked.connect(self.close)


# Do testowania
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    history_dialog = History()
    history_dialog.show()
    sys.exit(app.exec_())