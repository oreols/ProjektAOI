from PyQt5.QtWidgets import QDialog, QMessageBox, QInputDialog, QPushButton, QMenu, QLabel
from PyQt5.uic import loadUi
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QPixmap
from PyQt5.QtCore import Qt, QSortFilterProxyModel, QModelIndex, QTimer
import mysql.connector
from datetime import datetime
import os
import sys
import traceback
from db_config import DB_CONFIG

class History(QDialog):
    def __init__(self):
        super(History, self).__init__()
        
        try:
            # Wczytanie UI z pliku - sprawdzamy różne możliwe ścieżki
            ui_paths = ["ui/History.ui", "ui/history.ui"]
            
            ui_loaded = False
            for path in ui_paths:
                try:
                    if os.path.exists(path):
                        print(f"Ładowanie UI z: {path}")
                        loadUi(path, self)
                        self.ui_path = path
                        ui_loaded = True
                        print("UI załadowany pomyślnie")
                        break
                except Exception as ui_error:
                    print(f"Błąd ładowania UI z {path}: {ui_error}")
            
            if not ui_loaded:
                raise Exception(f"Nie można załadować pliku UI. Sprawdzone ścieżki: {ui_paths}")
            
            # Inicjalizacja bazy danych
            self.init_database()
            
            # Przygotowanie modelu danych
            self.model = QStandardItemModel(0, 3)  # 3 kolumny: Kod PCB, Data, Liczba komponentów
            self.model.setHorizontalHeaderLabels(['Kod PCB', 'Data analizy', 'Liczba komponentów'])
            
            # Przygotowanie modelu do filtrowania
            self.filter_proxy_model = QSortFilterProxyModel()
            self.filter_proxy_model.setSourceModel(self.model)
            self.filter_proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
            self.filter_proxy_model.setFilterKeyColumn(0)  # Filtruj po kodzie PCB
            
            # Ustawienie modelu dla tabeli
            self.tableWidget.setModel(self.filter_proxy_model)
            
            # Podłączenie pola wyszukiwania do filtra
            self.search_field.textChanged.connect(self.filter_proxy_model.setFilterFixedString)
            
            # Podłączenie przycisku odświeżania
            if hasattr(self, 'refreshButton'):
                print("Podłączanie przycisku odświeżania...")
                self.refreshButton.clicked.connect(self.refresh_data)
            else:
                print("UWAGA: Nie znaleziono przycisku 'refreshButton' w UI!")
                # Tworzymy przycisk programowo, jeśli nie ma go w UI
                self.refreshButton = QPushButton("Odśwież", self)
                self.refreshButton.setGeometry(150, self.height() - 45, 120, 30)
                self.refreshButton.clicked.connect(self.refresh_data)
                self.refreshButton.setStyleSheet("""
                    QPushButton {
                        background-color: #4f545c;
                        color: white;
                        font-size: 13px;
                        font-weight: 600;
                        padding: 6px 20px;
                        border-radius: 5px;
                        border: none;
                    }
                    QPushButton:hover { background-color: #5d6269; }
                    QPushButton:pressed { background-color: #6b6f79; }
                """)
                self.refreshButton.show()
            
            # Podłączenie przycisku usuwania
            if hasattr(self, 'deletePcbButton'):
                self.deletePcbButton.clicked.connect(self.delete_selected_pcb)
                # Domyślnie przycisk usuwania jest nieaktywny - wymaga zaznaczenia wiersza
                self.deletePcbButton.setEnabled(False)
            else:
                print("UWAGA: Nie znaleziono przycisku 'deletePcbButton' w UI!")
            
            # Reaguj na zaznaczenie wiersza (aby aktywować przycisk usuwania)
            self.tableWidget.selectionModel().selectionChanged.connect(self.on_selection_changed)
            
            # Dodaj kontekstowe menu do tabeli
            self.tableWidget.setContextMenuPolicy(Qt.CustomContextMenu)
            self.tableWidget.customContextMenuRequested.connect(self.show_context_menu)
            
            # Podłączenie podwójnego kliknięcia w tabeli
            self.tableWidget.doubleClicked.connect(self.show_pcb_details)
            
            # Wczytaj dane z bazy
            self.load_pcb_data()
            
            # Dodaj status odświeżania
            self.status_label = QLabel(self)
            self.status_label.setStyleSheet("color: #dcddde;")
            self.status_label.setText("")
            
            # Programowo utworzony refresh button jest zawsze na dole
            # Znajdujemy pozycję przycisku odświeżania i umieszczamy label obok niego
            if hasattr(self, 'refreshButton'):
                # Pobierz geometrię przycisku odświeżania
                refresh_btn_geo = self.refreshButton.geometry()
                # Ustaw label tuż obok przycisku
                self.status_label.setGeometry(refresh_btn_geo.right() + 10, refresh_btn_geo.top(), 300, refresh_btn_geo.height())
            else:
                # Domyślna pozycja jeśli przycisk odświeżania nie został znaleziony
                self.status_label.setGeometry(280, self.height() - 45, 300, 30)
            
            self.status_label.show()
            
        except Exception as e:
            print(f"Błąd podczas inicjalizacji okna historii: {e}")
            traceback.print_exc()
            QMessageBox.critical(None, "Błąd", f"Wystąpił błąd podczas inicjalizacji okna historii: {e}")
    
    def init_database(self):
        """Inicjalizacja połączenia z bazą danych MySQL"""
        try:
            print("Łączenie z bazą danych...")
            self.conn = mysql.connector.connect(**DB_CONFIG)
            self.cursor = self.conn.cursor()
            print("Połączono z bazą danych")
        except mysql.connector.Error as e:
            print(f"Błąd połączenia z bazą danych: {e}")
            QMessageBox.critical(self, "Błąd", f"Nie udało się połączyć z bazą danych: {e}")
            raise e
    
    def load_pcb_data(self):
        """Wczytuje dane PCB z bazy danych"""
        try:
            # Zapamietaj poprzedni filtr i zaznaczony wiersz
            current_filter = ""
            if self.search_field.text():
                current_filter = self.search_field.text()
                
            selected_pcb_code = None
            selected_indexes = self.tableWidget.selectedIndexes()
            if selected_indexes:
                source_index = self.filter_proxy_model.mapToSource(selected_indexes[0])
                selected_pcb_code = self.model.item(source_index.row(), 0).text()
            
            self.cursor.execute('''
                SELECT p.pcb_code, MAX(p.date_analyzed) as date_analyzed, COUNT(c.id) as component_count
                FROM pcb_records p
                LEFT JOIN components c ON p.pcb_code = c.pcb_code
                GROUP BY p.pcb_code
                ORDER BY date_analyzed DESC
            ''')
            
            records = self.cursor.fetchall()
            
            # Wyczyść model
            self.model.removeRows(0, self.model.rowCount())
            
            # Dodaj rekordy do modelu
            for record in records:
                pcb_code, date_analyzed, component_count = record
                date_str = date_analyzed.strftime('%Y-%m-%d %H:%M')
                
                row = [
                    QStandardItem(pcb_code),
                    QStandardItem(date_str),
                    QStandardItem(str(component_count))
                ]
                self.model.appendRow(row)
            
            # Przywróć filtr
            if current_filter:
                self.search_field.setText(current_filter)
            
            # Przywróć zaznaczenie wiersza
            if selected_pcb_code:
                for row in range(self.model.rowCount()):
                    if self.model.item(row, 0).text() == selected_pcb_code:
                        # Zaznacz ten wiersz
                        proxy_index = self.filter_proxy_model.mapFromSource(self.model.index(row, 0))
                        self.tableWidget.selectRow(proxy_index.row())
                        break
                
        except mysql.connector.Error as e:
            print(f"Błąd podczas wczytywania danych: {e}")
            QMessageBox.critical(self, "Błąd", f"Nie udało się wczytać danych: {e}")

    def show_pcb_details(self, index):
        """Wyświetla szczegóły wybranego PCB"""
        # Pobierz indeks z widoku proxy i przekształć go na indeks modelu bazowego
        source_index = self.filter_proxy_model.mapToSource(index)
        
        # Pobierz kod PCB z wybranej pozycji
        pcb_code = self.model.item(source_index.row(), 0).text()
        self.last_pcb_code = pcb_code  # Zapamiętaj wybrany kod PCB
        
        try:
            # Pobierz dane PCB
            self.cursor.execute('''
                SELECT p.image_path, p.date_analyzed
                FROM pcb_records p
                WHERE p.pcb_code = %s
            ''', (pcb_code,))
            
            pcb_data = self.cursor.fetchone()
            if not pcb_data:
                return
                
            image_path, date_analyzed = pcb_data
            
            # Pobierz komponenty
            try:
                self.cursor.execute('''
                    SELECT component_id, component_type, score, bbox, id
                    FROM components
                    WHERE pcb_code = %s
                    ORDER BY component_type, component_id
                ''', (pcb_code,))
                
                components = self.cursor.fetchall()
                self.last_components = components  # Zapamiętaj komponenty
            except mysql.connector.Error as e:
                # Jeśli kolumna component_type nie istnieje, spróbuj bez niej
                if "component_type" in str(e):
                    self.cursor.execute('''
                        SELECT component_id, "", score, bbox, id
                        FROM components
                        WHERE pcb_code = %s
                        ORDER BY component_id
                    ''', (pcb_code,))
                    components = self.cursor.fetchall()
                    self.last_components = components
                else:
                    raise e
            
            # Przygotuj szczegółowy komunikat
            details = f"PCB: {pcb_code}\n"
            details += f"Data analizy: {date_analyzed.strftime('%Y-%m-%d %H:%M')}\n"
            details += f"Liczba komponentów: {len(components)}\n\n"
            
            # Grupuj komponenty według typu
            components_by_type = {}
            for comp in components:
                component_id, component_type, score, bbox, db_id = comp
                if not component_type:
                    component_type = "Nieznany"
                
                if component_type not in components_by_type:
                    components_by_type[component_type] = []
                
                components_by_type[component_type].append((component_id, score, bbox, db_id))
            
            # Wyświetl komponenty pogrupowane według typu
            details += "Wykryte komponenty:\n"
            for component_type, comps in components_by_type.items():
                details += f"\n== {component_type} ({len(comps)}) ==\n"
                for comp_id, score, bbox, db_id in comps:
                    details += f"- {comp_id} (Score: {score:.2f})\n"
            
            # Wyświetl szczegóły
            QMessageBox.information(self, "Szczegóły PCB", details)
            
        except mysql.connector.Error as e:
            QMessageBox.critical(self, "Błąd", f"Nie udało się pobrać szczegółów: {e}")
            
    def show_context_menu(self, position):
        """Wyświetla menu kontekstowe po kliknięciu prawym przyciskiem myszy na tabeli"""
        index = self.tableWidget.indexAt(position)
        if not index.isValid():
            return
            
        menu = QMenu(self)
        view_action = menu.addAction("Pokaż szczegóły")
        delete_action = menu.addAction("Usuń płytkę PCB")
        
        action = menu.exec_(self.tableWidget.mapToGlobal(position))
        
        if action == view_action:
            self.show_pcb_details(index)
        elif action == delete_action:
            self.delete_pcb(index)

    def delete_selected_pcb(self):
        """Wywołuje funkcję delete_pcb dla aktualnie zaznaczonego wiersza"""
        selected_indexes = self.tableWidget.selectedIndexes()
        if selected_indexes:
            self.delete_pcb(selected_indexes[0])
        else:
            QMessageBox.warning(self, "Uwaga", "Najpierw wybierz płytkę PCB do usunięcia.")

    def delete_pcb(self, index):
        """Usuwa całą płytkę PCB wraz z jej komponentami z bazy danych"""
        # Sprawdź czy index jest obiektem QModelIndex
        if not isinstance(index, QModelIndex):
            QMessageBox.warning(self, "Uwaga", "Wystąpił błąd z indeksem. Wybierz płytkę ponownie.")
            return
            
        # Pobierz indeks z widoku proxy i przekształć go na indeks modelu bazowego
        source_index = self.filter_proxy_model.mapToSource(index)
        
        # Pobierz kod PCB z wybranej pozycji
        pcb_code = self.model.item(source_index.row(), 0).text()
        
        # Potwierdź usunięcie
        confirm = QMessageBox.question(
            self, "Potwierdź usunięcie",
            f"Czy na pewno chcesz usunąć płytkę PCB: {pcb_code} i wszystkie jej komponenty?\n\nTa operacja jest nieodwracalna!",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No  # Domyślna opcja to "Nie" - dla bezpieczeństwa
        )
        
        if confirm == QMessageBox.Yes:
            try:
                # Pobierz ścieżkę obrazu przed usunięciem rekordu
                self.cursor.execute("SELECT image_path FROM pcb_records WHERE pcb_code = %s", (pcb_code,))
                result = self.cursor.fetchone()
                image_path = result[0] if result else None
                
                # Najpierw usuń komponenty (ze względu na klucz obcy)
                self.cursor.execute("DELETE FROM components WHERE pcb_code = %s", (pcb_code,))
                rows_deleted = self.cursor.rowcount
                
                # Następnie usuń rekord płytki PCB
                self.cursor.execute("DELETE FROM pcb_records WHERE pcb_code = %s", (pcb_code,))
                
                self.conn.commit()
                
                # Usuń plik obrazu jeśli istnieje
                if image_path and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        print(f"Usunięto plik obrazu: {image_path}")
                    except Exception as e:
                        print(f"Błąd podczas usuwania pliku obrazu: {e}")
                
                QMessageBox.information(
                    self, "Sukces", 
                    f"Płytka PCB '{pcb_code}' została usunięta.\n"
                    f"Liczba usuniętych komponentów: {rows_deleted}"
                )
                
                # Odśwież dane
                self.load_pcb_data()
                
                # Dezaktywuj przycisk usuwania
                self.deletePcbButton.setEnabled(False)
                
            except mysql.connector.Error as e:
                self.conn.rollback()
                QMessageBox.critical(self, "Błąd", f"Nie udało się usunąć płytki PCB: {e}")

    def delete_component(self):
        """Usuwa wybrany komponent z bazy danych"""
        if not hasattr(self, 'last_pcb_code') or not hasattr(self, 'last_components') or not self.last_components:
            QMessageBox.warning(self, "Uwaga", "Najpierw wybierz płytkę PCB, aby zobaczyć jej komponenty.")
            return
            
        # Pobierz typ komponentu do usunięcia
        component_types = set()
        for comp in self.last_components:
            _, comp_type, _, _, _ = comp
            if comp_type:
                component_types.add(comp_type)
            else:
                component_types.add("Nieznany")
                
        if not component_types:
            QMessageBox.warning(self, "Uwaga", "Brak komponentów do usunięcia.")
            return
            
        # Lista typów komponentów do wyboru
        type_options = sorted(list(component_types))
        selected_type, ok = QInputDialog.getItem(
            self, "Wybierz typ komponentu", "Typ komponentu do usunięcia:", 
            type_options, 0, False
        )
        
        if not ok or not selected_type:
            return
            
        # Filtruj komponenty według wybranego typu
        components_of_type = []
        for comp in self.last_components:
            comp_id, comp_type, score, bbox, db_id = comp
            if (comp_type == selected_type) or (not comp_type and selected_type == "Nieznany"):
                components_of_type.append((comp_id, db_id))
                
        if not components_of_type:
            QMessageBox.warning(self, "Uwaga", f"Brak komponentów typu {selected_type}.")
            return
            
        # Wybierz konkretny komponent do usunięcia
        component_options = [f"{comp_id} (ID: {db_id})" for comp_id, db_id in components_of_type]
        selected_comp, ok = QInputDialog.getItem(
            self, "Wybierz komponent", f"Komponent typu {selected_type} do usunięcia:", 
            component_options, 0, False
        )
        
        if not ok or not selected_comp:
            return
            
        # Pobierz ID komponentu z bazy
        selected_db_id = int(selected_comp.split("(ID: ")[1].split(")")[0])
        
        # Potwierdź usunięcie
        confirm = QMessageBox.question(
            self, "Potwierdź usunięcie",
            f"Czy na pewno chcesz usunąć komponent:\n{selected_comp}?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                # Usuń komponent
                self.cursor.execute("DELETE FROM components WHERE id = %s", (selected_db_id,))
                self.conn.commit()
                
                QMessageBox.information(self, "Sukces", "Komponent został usunięty.")
                
                # Odśwież dane
                self.load_pcb_data()
                
                # Ponownie wyświetl szczegóły PCB, jeśli były pokazane
                if hasattr(self, 'last_pcb_code'):
                    pcb_code = self.last_pcb_code
                    
                    # Znajdź indeks wiersza z tym kodem PCB
                    for row in range(self.model.rowCount()):
                        if self.model.item(row, 0).text() == pcb_code:
                            # Utwórz indeks i wywołaj pokazanie szczegółów
                            index = self.model.index(row, 0)
                            proxy_index = self.filter_proxy_model.mapFromSource(index)
                            self.show_pcb_details(proxy_index)
                            break
                
            except mysql.connector.Error as e:
                QMessageBox.critical(self, "Błąd", f"Nie udało się usunąć komponentu: {e}")
            
    def on_selection_changed(self, selected, deselected):
        """Aktywuje/dezaktywuje przycisk usuwania w zależności od zaznaczenia"""
        if hasattr(self, 'deletePcbButton'):
            self.deletePcbButton.setEnabled(len(selected.indexes()) > 0)

    def closeEvent(self, event):
        """Zamykanie okna i bazy danych"""
        if hasattr(self, 'conn'):
            self.conn.close()
        event.accept()

    def refresh_data(self):
        """Odświeża dane z bazy"""
        try:
            print("Rozpoczynam odświeżanie danych...")
            self.status_label.setText("")
            QTimer.singleShot(100, self._perform_refresh)  # Użyj QTimer, aby dać czas na aktualizację interfejsu
        except Exception as e:
            print(f"Błąd podczas odświeżania: {e}")
            traceback.print_exc()
            self.status_label.setText(f"Błąd odświeżania: {str(e)[:30]}...")
            QMessageBox.critical(self, "Błąd", f"Wystąpił błąd podczas odświeżania danych: {e}")
            
    def _perform_refresh(self):
        """Wykonuje właściwe odświeżanie danych"""
        try:
            # Explicite zamykamy i otwieramy połączenie
            if hasattr(self, 'conn'):
                try:
                    if self.conn.is_connected():
                        self.conn.close()
                        print("Zamknięto poprzednie połączenie")
                except Exception as close_error:
                    print(f"Błąd przy zamykaniu połączenia: {close_error}")
            
            # Inicjalizuj ponownie połączenie
            self.init_database()
            
            # Wczytaj dane
            self.load_pcb_data()
            
            # Aktualizuj status (usunięto tekst)
            self.status_label.setText("")
            
            # Usunięto komunikat QMessageBox.information - nie wyświetlamy komunikatu o odświeżeniu
            print("Dane odświeżone pomyślnie")
        except Exception as e:
            print(f"Błąd podczas odświeżania: {e}")
            traceback.print_exc()
            self.status_label.setText(f"Błąd odświeżania: {str(e)[:30]}...")
            QMessageBox.critical(self, "Błąd", f"Wystąpił błąd podczas odświeżania danych: {e}")

# Do testowania
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    history_dialog = History()
    history_dialog.show()
    sys.exit(app.exec_())