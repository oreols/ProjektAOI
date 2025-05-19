from PyQt5.QtWidgets import (QDialog, QDateEdit, QFileDialog,
                          QMessageBox, QComboBox, QVBoxLayout, QLabel, QHBoxLayout,
                          QPushButton, QFrame, QHeaderView, QTableView,
                          QApplication, QAbstractItemView, QSizePolicy, QSpacerItem, QToolButton)
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QDate, pyqtSlot, QTimer, QDateTime, QModelIndex, QSize
from PyQt5.QtGui import QPixmap, QFont, QColor, QStandardItemModel, QStandardItem, QIcon, QPen

import mysql.connector
from datetime import datetime
import os
import sys
import traceback
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as reportlab_colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# Usunięto: from reportlab.pdfbase import pdfmetrics
# Usunięto: from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import inch, cm
from db_config import DB_CONFIG
import pkg_resources

# Usunięto sekcję FONT_DIR, DEJAVU_SANS_PATH i logikę rejestracji czcionek DejaVu.
# Domyślne czcionki ReportLab (Helvetica) będą używane.
PDF_DEFAULT_FONT = 'Helvetica'
PDF_DEFAULT_FONT_BOLD = 'Helvetica-Bold'
PDF_DEFAULT_FONT_ITALIC = 'Helvetica-Oblique'

class Reports(QDialog):
    def __init__(self, user_id=None):
        super(Reports, self).__init__()

        self.user_id = user_id
        self.user_data = None
        self.tableModel = QStandardItemModel(self)
        self.current_pcb_code = None
        self.current_view_mode = "date"

        self.conn = None
        self.cursor = None

        try:
            loadUi("ui/report.ui", self)
        except Exception as e:
            print(f"Nie udało się załadować ui/report.ui: {e}. UI zostanie utworzone programowo.")
            if not hasattr(self, 'mainVerticalLayout'):
                 self.mainVerticalLayout = QVBoxLayout(self)
                 self.setLayout(self.mainVerticalLayout)
        try:
            self.init_database_connection()
            self.configure_ui_elements_modern()

            if self.user_id:
                self.user_data = self.get_user_data()

            self.setup_dynamic_elements()
            self.load_report_codes()
            self.load_report_data()

        except Exception as e:
            print(f"Błąd inicjalizacji raportu: {e}")
            traceback.print_exc()
            QMessageBox.critical(None, "Błąd Krytyczny", f"Wystąpił błąd podczas inicjalizacji: {e}")

    def configure_ui_elements_modern(self):
        # ... (BEZ ZMIAN - Konfiguracja UI i Stylesheet) ...
        self.setWindowTitle("Raporty Analizy PCB")
        self.setMinimumSize(1000, 800)
        self.setObjectName("ReportsDialog")
        self.setStyleSheet("""
            QDialog#ReportsDialog { background-color: #2f3136; color: #e8e9ea; font-family: "Segoe UI", Roboto, Cantarell, "Helvetica Neue", sans-serif; }
            QLabel { color: #b9bbbe; font-size: 14px; padding-top: 5px; }
            QFrame#titleFrame { background-color: #36393f; border: 1px solid #202225; padding: 0px 15px; min-height: 50px; max-height: 50px; border-radius: 8px;}
            QLabel#titleLabel { font-size: 18px; font-weight: 600; color: #ffffff; padding: 0px; background-color: transparent; border: none; }
            QLabel#currentDateTimeLabel { font-size: 14px; color: #96989d; padding-left: 15px; }
            QDateEdit, QComboBox { background-color: #40444b; color: #dcddde; border: 1px solid #282a2e; border-radius: 5px; padding: 9px 12px; min-height: 20px; font-size: 14px;}
            QDateEdit:focus, QComboBox:focus { border-color: #5865F2; }
            QDateEdit::drop-down, QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 22px; border-left: 1px solid #282a2e;}
            QComboBox QAbstractItemView { background-color: #36393f; border: 1px solid #282a2e; selection-background-color: #5865F2; color: #dcddde; padding: 4px; outline: none;}
            QCalendarWidget { background-color: #2f3136; color: #dcddde; }
            QCalendarWidget QToolButton { background-color: #40444b; color: white; border: 1px solid #282a2e; border-radius: 4px; padding: 6px; margin: 2px; font-size: 13px; font-weight: 500; min-width: 25px; }
            QCalendarWidget QToolButton:hover { background-color: #4f545c; }
            QCalendarWidget QToolButton:pressed { background-color: #5865F2; }
            QCalendarWidget QMenu { background-color: #40444b; color: #dcddde; border: 1px solid #282a2e; padding: 5px;}
            QCalendarWidget QSpinBox { background-color: #40444b; color: #dcddde; border: 1px solid #282a2e; padding: 5px; border-radius: 4px; font-size: 13px; margin: 2px;}
            QCalendarWidget QTableView { gridline-color: #40444b; background-color: #36393f; selection-background-color: #5865F2; selection-color: white; border-radius: 4px; color: #dcddde;}
            QCalendarWidget QAbstractItemView:disabled { color: #72767d; }
            QCalendarWidget #qt_calendar_navigationbar { background-color: #2f3136; border: none; padding: 4px;}
            QFrame#reportContentFrame { background-color: #36393f; border-radius: 8px; border: 1px solid #202225; }
            QTableView#mainTableView { background-color: transparent; border: none; border-radius: 0px; gridline-color: #40444b; color: #dcddde; font-size: 14px; selection-background-color: #5865F2; selection-color: #ffffff; outline: none; alternate-background-color: #3a3e44; }
            QTableView#mainTableView::item { padding: 10px 12px; border-bottom: 1px solid #40444b; border-right: 1px solid #40444b;}
            QTableView#mainTableView::item:selected { background-color: #5865F2; color: #ffffff; }
            QTableView#mainTableView::item:hover:!selected { background-color: #4f545c; }
            QHeaderView::section { background-color: #2a2d31; color: #ffffff; padding: 12px 10px; border: none; border-bottom: 1px solid #222427; font-size: 14px; font-weight: 600;}
            QHeaderView::section:horizontal { border-right: 1px solid #222427; }
            QHeaderView::section:horizontal:last { border-right: none; }
            QTableView#mainTableView QTableCornerButton::section { background-color: #2a2d31; border: none; border-bottom: 1px solid #222427; border-right: 1px solid #222427;}
            QPushButton { background-color: #5865F2; color: white; font-size: 14px; font-weight: 600; padding: 9px 18px; border-radius: 5px; border: none; min-height: 20px; outline: none;}
            QPushButton:hover { background-color: #4e5dcf; }
            QPushButton:pressed { background-color: #404ab8; }
            QPushButton#refreshButton, QPushButton#showAllButton { background-color: #4f545c; padding: 9px 15px; }
            QPushButton#refreshButton:hover, QPushButton#showAllButton:hover { background-color: #5d6269; }
            QPushButton#refreshButton:pressed, QPushButton#showAllButton:pressed { background-color: #6b6f79; }
            QScrollBar:vertical { border: none; background: #2f3136; width: 14px; margin: 0px; }
            QScrollBar::handle:vertical { background: #4f545c; min-height: 30px; border-radius: 7px; }
            QScrollBar::handle:vertical:hover { background: #5a606b; }
            QScrollBar:horizontal { border: none; background: #2f3136; height: 14px; margin: 0px; }
            QScrollBar::handle:horizontal { background: #4f545c; min-width: 30px; border-radius: 7px; }
            QScrollBar::handle:horizontal:hover { background: #5a606b; }
        """)
        
        if not hasattr(self, 'mainVerticalLayout') or self.layout() != self.mainVerticalLayout:
            current_layout = self.layout()
            if current_layout is not None:
                while current_layout.count():
                    item = current_layout.takeAt(0)
                    widget = item.widget(); layout_item = item.layout()
                    if widget: widget.deleteLater()
                    elif layout_item:
                        while layout_item.count():
                            sub_item = layout_item.takeAt(0)
                            if sub_item.widget(): sub_item.widget().deleteLater()
                        layout_item.deleteLater()
                current_layout.deleteLater()
            self.mainVerticalLayout = QVBoxLayout(self)
            self.setLayout(self.mainVerticalLayout)
        while self.mainVerticalLayout.count():
            child = self.mainVerticalLayout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            elif child.layout():
                temp_layout = child.layout()
                while temp_layout.count():
                    temp_item = temp_layout.takeAt(0)
                    if temp_item.widget(): temp_item.widget().deleteLater()
                    elif temp_item.layout():
                         inner_layout = temp_item.layout()
                         while inner_layout.count(): inner_item = inner_layout.takeAt(0)
                         if inner_item.widget(): inner_item.widget().deleteLater()
                         inner_layout.deleteLater()
                temp_layout.deleteLater()
        self.mainVerticalLayout.setContentsMargins(20, 20, 20, 20)
        self.mainVerticalLayout.setSpacing(18)
        headerOuterLayout = QHBoxLayout()
        headerOuterLayout.setSpacing(0) 
        headerOuterLayout.setAlignment(Qt.AlignVCenter) 
        self.titleFrame = QFrame(self) 
        self.titleFrame.setObjectName("titleFrame") 
        titleFrameLayout = QHBoxLayout(self.titleFrame) 
        titleFrameLayout.setContentsMargins(0,0,0,0) 
        self.titleLabel = QLabel("Raporty Analizy PCB", self.titleFrame)
        self.titleLabel.setObjectName("titleLabel")
        self.titleLabel.setAlignment(Qt.AlignCenter) 
        titleFrameLayout.addWidget(self.titleLabel)
        headerOuterLayout.addWidget(self.titleFrame, 1) 
        self.currentDateTimeLabel = QLabel(self)
        self.currentDateTimeLabel.setObjectName("currentDateTimeLabel")
        self.currentDateTimeLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.currentDateTimeLabel.setFixedHeight(50) 
        headerOuterLayout.addWidget(self.currentDateTimeLabel) 
        self.mainVerticalLayout.addLayout(headerOuterLayout)
        controlsLayout = QHBoxLayout()
        controlsLayout.setSpacing(10)
        date_label = QLabel("Data raportu:", self)
        self.dateEdit = QDateEdit(self)
        self.dateEdit.setCalendarPopup(True)
        self.dateEdit.setFixedWidth(140)
        code_label = QLabel("Kod PCB:", self)
        self.reportCodeCombo = QComboBox(self)
        self.reportCodeCombo.setMinimumWidth(250)
        self.reportCodeCombo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.showAllButton = QPushButton("Wszystkie PCB", self)
        self.showAllButton.setObjectName("showAllButton")
        self.refreshButton = QPushButton("Odśwież", self)
        self.refreshButton.setObjectName("refreshButton")
        self.dateEdit.blockSignals(True)
        self.dateEdit.setDate(QDate.currentDate())
        self.dateEdit.blockSignals(False)
        self.dateEdit.dateChanged.connect(self.on_date_changed)
        self.reportCodeCombo.currentIndexChanged.connect(self.on_report_code_changed)
        self.showAllButton.clicked.connect(self.load_all_pcb_data)
        self.refreshButton.clicked.connect(self.refresh_data)
        controlsLayout.addWidget(date_label); controlsLayout.addWidget(self.dateEdit)
        controlsLayout.addSpacing(15); controlsLayout.addWidget(code_label)
        controlsLayout.addWidget(self.reportCodeCombo, 1)
        controlsLayout.addSpacerItem(QSpacerItem(15, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        controlsLayout.addWidget(self.showAllButton); controlsLayout.addWidget(self.refreshButton)
        self.mainVerticalLayout.addLayout(controlsLayout)
        self.reportFrameTable = QFrame(self) 
        self.reportFrameTable.setObjectName("reportContentFrame") 
        reportFrameLayoutTable = QVBoxLayout(self.reportFrameTable)
        reportFrameLayoutTable.setContentsMargins(3, 3, 3, 3) 
        self.tableView = QTableView(self.reportFrameTable)
        self.tableView.setObjectName("mainTableView")
        self.tableView.setModel(self.tableModel)
        self.tableView.setAlternatingRowColors(True); self.tableView.setShowGrid(True)
        self.tableView.setWordWrap(True); self.tableView.verticalHeader().setVisible(False)
        self.tableView.verticalHeader().setDefaultSectionSize(45)
        self.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableView.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableView.horizontalHeader().setStretchLastSection(True)
        self.tableView.doubleClicked.connect(self.on_table_double_clicked)
        reportFrameLayoutTable.addWidget(self.tableView)
        self.mainVerticalLayout.addWidget(self.reportFrameTable, 1)
        footerLayout = QHBoxLayout()
        self.generatePdfButton = QPushButton("Generuj raport PDF", self)
        self.generatePdfButton.setObjectName("generatePdfButton")
        self.generatePdfButton.clicked.connect(self.generate_pdf_report)
        footerLayout.addStretch(1); footerLayout.addWidget(self.generatePdfButton)
        self.mainVerticalLayout.addLayout(footerLayout)

    def setup_dynamic_elements(self):
        # ... (BEZ ZMIAN) ...
        self.dateTimeTimer = QTimer(self)
        self.dateTimeTimer.timeout.connect(self.update_current_datetime_label)
        self.dateTimeTimer.start(1000)
        self.update_current_datetime_label()

    def update_current_datetime_label(self):
        # ... (BEZ ZMIAN) ...
        if hasattr(self, 'currentDateTimeLabel'):
            now = QDateTime.currentDateTime()
            self.currentDateTimeLabel.setText(now.toString("dd.MM.yyyy HH:mm"))

    def _close_db_connection(self):
        # ... (BEZ ZMIAN) ...
        if hasattr(self, 'cursor') and self.cursor:
            try:
                self.cursor.close()
            except Exception as e:
                print(f"Błąd podczas zamykania kursora: {e}")
            self.cursor = None
        if hasattr(self, 'conn') and self.conn and self.conn.is_connected():
            try:
                self.conn.close()
            except Exception as e:
                print(f"Błąd podczas zamykania połączenia z bazą: {e}")
            self.conn = None

    def init_database_connection(self):
        # ... (BEZ ZMIAN) ...
        self._close_db_connection() 
        try:
            self.conn = mysql.connector.connect(**DB_CONFIG)
            self.conn.autocommit = True 
            self.cursor = self.conn.cursor(dictionary=True)
        except mysql.connector.Error as e:
            QMessageBox.critical(self, "Błąd Bazy Danych", f"Nie można połączyć się z bazą danych: {e}")
            self.conn = None 
            self.cursor = None
            raise e

    def get_user_data(self):
        # ... (BEZ ZMIAN) ...
        try:
            if not self.conn or not self.conn.is_connected(): self.init_database_connection()
            self.cursor.execute("SELECT u.id, u.name, u.surname, u.email, p.name AS position_name FROM user u LEFT JOIN `position` p ON u.position_id = p.id WHERE u.id = %s", (self.user_id,))
            return self.cursor.fetchone()
        except mysql.connector.Error as e:
            print(f"Błąd pobierania danych użytkownika: {e}")
            QMessageBox.warning(self, "Błąd Danych", "Nie udało się pobrać danych użytkownika.")
            return None
        except AttributeError: 
             QMessageBox.warning(self, "Błąd Połączenia", "Brak aktywnego połączenia z bazą danych (get_user_data).")
             return None

    def load_report_codes(self):
        # ... (BEZ ZMIAN) ...
        try:
            if not self.conn or not self.conn.is_connected(): self.init_database_connection()
            self.cursor.execute("SELECT pcb_code, date_analyzed FROM pcb_records ORDER BY date_analyzed DESC")
            records = self.cursor.fetchall()
            
            if not hasattr(self, 'reportCodeCombo'): return 

            current_selection_data = self.reportCodeCombo.currentData()
            self.reportCodeCombo.blockSignals(True)
            self.reportCodeCombo.clear()
            self.reportCodeCombo.addItem("-- Wybierz kod PCB --", None)
            
            idx_to_select = 0
            for i, record in enumerate(records):
                display_text = f"{record['pcb_code']} (analiza: {record['date_analyzed'].strftime('%y-%m-%d %H:%M')})"
                self.reportCodeCombo.addItem(display_text, record['pcb_code'])
                if record['pcb_code'] == current_selection_data:
                    idx_to_select = i + 1
            
            self.reportCodeCombo.setCurrentIndex(idx_to_select)
            self.reportCodeCombo.blockSignals(False)
            
            if self.reportCodeCombo.currentIndex() == 0 and current_selection_data is not None:
                self.current_pcb_code = None
            elif self.reportCodeCombo.currentIndex() > 0:
                 self.current_pcb_code = self.reportCodeCombo.currentData()

        except mysql.connector.Error as e:
            QMessageBox.warning(self, "Błąd Ładowania Kodów", f"Nie można załadować kodów PCB: {e}")
        except AttributeError:
             QMessageBox.warning(self, "Błąd Połączenia", "Brak aktywnego połączenia z bazą danych (load_report_codes).")

    def on_date_changed(self):
        # ... (BEZ ZMIAN) ...
        if not hasattr(self, 'reportCodeCombo') or not hasattr(self, 'dateEdit'): return
        self.dateEdit.setStyleSheet("") 
        self.current_pcb_code = None
        self.current_view_mode = "date"
        if self.reportCodeCombo.currentIndex() != 0:
            self.reportCodeCombo.setCurrentIndex(0) 
        else: 
            if hasattr(self, 'titleLabel'):
                 self.titleLabel.setText(f"Raporty z dnia: {self.dateEdit.date().toString('dd.MM.yyyy')}")
            self.load_report_data()

    def on_report_code_changed(self, index):
        # ... (BEZ ZMIAN) ...
        if not hasattr(self, 'reportCodeCombo') or not hasattr(self, 'dateEdit'): return
        self.dateEdit.setStyleSheet("")
        pcb_code_data = self.reportCodeCombo.itemData(index)
        if pcb_code_data:
            self.current_pcb_code = pcb_code_data
            self.display_pcb_components(pcb_code_data) 
        else: 
            self.current_pcb_code = None
            if hasattr(self, 'titleLabel') and hasattr(self, 'dateEdit'):
                 self.titleLabel.setText(f"Raporty z dnia: {self.dateEdit.date().toString('dd.MM.yyyy')}")
            self.load_report_data() 

    def load_all_pcb_data(self):
        # ... (BEZ ZMIAN) ...
        try:
            if not self.conn or not self.conn.is_connected(): self.init_database_connection()
            self.cursor.execute("""
                SELECT p.pcb_code, p.date_analyzed, COUNT(c.id) as component_count
                FROM pcb_records p LEFT JOIN components c ON p.pcb_code = c.pcb_code
                GROUP BY p.pcb_code, p.date_analyzed ORDER BY p.date_analyzed DESC
            """)
            records = self.cursor.fetchall()
            self.display_report_data_in_table(records)
            
            if hasattr(self, 'titleLabel'):
                self.titleLabel.setText(f"Wszystkie Zarejestrowane PCB ({len(records)})")
            
            self.current_pcb_code = None
            self.current_view_mode = "all" 
            if hasattr(self, 'reportCodeCombo'): self.reportCodeCombo.setCurrentIndex(0)
            if hasattr(self, 'dateEdit'):
                self.dateEdit.setStyleSheet("QDateEdit { background-color: #3a3e44; color: #72767d; }")
        except mysql.connector.Error as e:
            QMessageBox.critical(self, "Błąd Ładowania Danych", f"Nie można załadować wszystkich danych PCB: {e}")
        except AttributeError:
             QMessageBox.warning(self, "Błąd Połączenia", "Brak aktywnego połączenia z bazą danych (load_all_pcb_data).")

    def display_pcb_components(self, pcb_code):
        # ... (BEZ ZMIAN - metoda GUI pozostaje nietknięta) ...
        try:
            if not self.conn or not self.conn.is_connected(): self.init_database_connection()
            self.cursor.execute("SELECT component_id, component_type, score FROM components WHERE pcb_code = %s ORDER BY component_type, component_id", (pcb_code,))
            components = self.cursor.fetchall()
            self.tableModel.clear()
            
            if not hasattr(self.tableView, 'horizontalHeader') or not hasattr(self, 'titleLabel'): return

            if not components:
                self.tableModel.setHorizontalHeaderLabels(['Informacja'])
                self.tableModel.appendRow([QStandardItem(f"Brak komponentów dla PCB: {pcb_code}")])
                self.tableView.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
                self.titleLabel.setText(f"Analiza PCB: {pcb_code} (0 komponentów)")
                self.current_view_mode = "components" 
                return

            self.tableModel.setHorizontalHeaderLabels(['ID Komponentu', 'Typ', 'Wynik'])
            self.tableView.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            self.tableView.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            self.tableView.setColumnWidth(1, 180) 
            self.tableView.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
            self.tableView.setColumnWidth(2, 100)

            for comp in components:
                comp_id_text = comp['component_id']
                type_text = comp['component_type'] if comp['component_type'] else "N/A"
                score_text = f"{comp['score']:.2f}"
                bg_color, text_color = None, None 
                if comp['score'] < 0.5: 
                    bg_color, text_color = QColor("#c62828"), Qt.white
                    comp_id_text = f"❌ {comp_id_text}" 
                elif comp['score'] < 0.8: 
                    bg_color, text_color = QColor(85, 85, 85, 200), QColor(255, 235, 59)   
                    comp_id_text = f"⚠ {comp_id_text}" 
                
                id_item = QStandardItem(comp_id_text)
                type_item = QStandardItem(type_text)
                score_item = QStandardItem(score_text)
                score_item.setTextAlignment(Qt.AlignCenter)
                row_items = [id_item, type_item, score_item]
                for item in row_items:
                    if bg_color: item.setBackground(bg_color)
                    if text_color: item.setForeground(text_color)
                    else: 
                        item.setBackground(QColor(Qt.transparent)) 
                        item.setForeground(QColor("#dcddde")) 
                self.tableModel.appendRow(row_items)
            
            self.titleLabel.setText(f"Analiza PCB: {pcb_code} ({len(components)} komponentów)")
            self.current_view_mode = "components"
        except mysql.connector.Error as e: QMessageBox.critical(self, "Błąd Wyświetlania", f"Nie można wyświetlić komponentów: {e}")
        except AttributeError: QMessageBox.warning(self, "Błąd Połączenia", "Brak aktywnego połączenia z bazą danych (display_pcb_components).")

    def load_report_data(self):
        # ... (BEZ ZMIAN) ...
        try:
            if not self.conn or not self.conn.is_connected(): self.init_database_connection()
            if not hasattr(self, 'dateEdit'): return

            selected_date_str = self.dateEdit.date().toString("yyyy-MM-dd")
            self.cursor.execute("""
                SELECT p.pcb_code, p.date_analyzed, COUNT(c.id) as component_count
                FROM pcb_records p LEFT JOIN components c ON p.pcb_code = c.pcb_code
                WHERE DATE(p.date_analyzed) = %s
                GROUP BY p.pcb_code, p.date_analyzed ORDER BY p.date_analyzed DESC
            """, (selected_date_str,))
            records = self.cursor.fetchall()
            self.display_report_data_in_table(records)
            
            if not self.current_pcb_code: 
                if hasattr(self, 'titleLabel'):
                    self.titleLabel.setText(f"Raporty z dnia: {self.dateEdit.date().toString('dd.MM.yyyy')}")
                self.current_view_mode = "date" 

        except mysql.connector.Error as e: QMessageBox.critical(self, "Błąd Ładowania Danych", f"Nie można załadować danych raportu: {e}")
        except AttributeError: QMessageBox.warning(self, "Błąd Połączenia", "Brak aktywnego połączenia z bazą danych (load_report_data).")

    def display_report_data_in_table(self, records):
        # ... (BEZ ZMIAN) ...
        self.tableModel.clear()
        if not hasattr(self.tableView, 'horizontalHeader'): return
        if not records:
            self.tableModel.setHorizontalHeaderLabels(['Informacja'])
            self.tableModel.appendRow([QStandardItem("Brak danych do wyświetlenia.")])
            self.tableView.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            return
        self.tableModel.setHorizontalHeaderLabels(['Kod PCB', 'Data Analizy', 'Liczba Komponentów'])
        self.tableView.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tableView.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tableView.setColumnWidth(1, 180) 
        self.tableView.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tableView.setColumnWidth(2, 160) 
        for record in records:
            pcb_item = QStandardItem(record['pcb_code'])
            pcb_item.setData(record['pcb_code'], Qt.UserRole)
            date_item = QStandardItem(record['date_analyzed'].strftime('%Y-%m-%d %H:%M:%S'))
            date_item.setTextAlignment(Qt.AlignCenter)
            count_item = QStandardItem(str(record['component_count']))
            count_item.setTextAlignment(Qt.AlignCenter)
            self.tableModel.appendRow([pcb_item, date_item, count_item])

    def on_table_double_clicked(self, index: QModelIndex):
        # ... (BEZ ZMIAN) ...
        if not index.isValid(): return
        current_headers = [self.tableModel.horizontalHeaderItem(i).text() for i in range(self.tableModel.columnCount()) if self.tableModel.horizontalHeaderItem(i)]
        if 'Kod PCB' in current_headers:
            pcb_code = self.tableModel.item(index.row(), 0).data(Qt.UserRole)
            if pcb_code and hasattr(self, 'reportCodeCombo'):
                combo_idx = self.reportCodeCombo.findData(pcb_code)
                if combo_idx != -1: self.reportCodeCombo.setCurrentIndex(combo_idx) 
                else:
                    self.current_pcb_code = pcb_code
                    self.display_pcb_components(pcb_code)
        elif 'ID Komponentu' in current_headers:
            full_id_text = self.tableModel.item(index.row(), 0).text()
            comp_id_actual = full_id_text.split(" ", 1)[1] if " " in full_id_text else full_id_text 
            comp_type = self.tableModel.item(index.row(), 1).text()
            comp_score = self.tableModel.item(index.row(), 2).text()
            QMessageBox.information(self, "Szczegóły Komponentu", f"PCB: {self.current_pcb_code}\nID: {comp_id_actual}\nTyp: {comp_type}\nWynik: {comp_score}")

    
    def generate_pdf_report(self):
        if not self.current_pcb_code:
            QMessageBox.warning(self, self._transliterate("Wybierz PCB"), self._transliterate("Najpierw wybierz kod PCB z listy."))
            return

        file_path, _ = QFileDialog.getSaveFileName(self, self._transliterate("Zapisz raport PDF"),
                                                 f"Raport_PCB_{self.current_pcb_code}.pdf", "Pliki PDF (*.pdf)")
        if not file_path: return
        if not file_path.lower().endswith('.pdf'): file_path += '.pdf'

        try:
            doc = SimpleDocTemplate(file_path, pagesize=A4,
                                    topMargin=1.5 * cm, bottomMargin=2.0 * cm,
                                    leftMargin=1.5 * cm, rightMargin=1.5 * cm)
            
            styles = getSampleStyleSheet()
            
            # --- Definicje stylów dla PDF ---
            title_style = ParagraphStyle(
                name='ReportTitle', fontName=PDF_DEFAULT_FONT_BOLD, fontSize=18, alignment=1,
                spaceAfter=0.5 * cm, textColor=reportlab_colors.HexColor("#333333"))
            
            subtitle_style = ParagraphStyle(
                name='ReportSubtitle', fontName=PDF_DEFAULT_FONT, fontSize=12, alignment=1,
                spaceAfter=0.7 * cm, textColor=reportlab_colors.HexColor("#444444"))
            
            section_heading_style = ParagraphStyle(
                name='SectionHeading', fontName=PDF_DEFAULT_FONT_BOLD, fontSize=13,
                spaceBefore=0.7 * cm, spaceAfter=0.3 * cm, textColor=reportlab_colors.HexColor("#2c3e50"),
                borderBottomWidth=0.5, borderBottomColor=reportlab_colors.HexColor("#bbbbbb"),
                leftIndent=0, rightIndent=0, keepWithNext=1
            )
            
            table_text_normal_style = ParagraphStyle(
                name='TableTextNormal', fontName=PDF_DEFAULT_FONT, fontSize=9, leading=11,
                textColor=reportlab_colors.black)
            
            table_text_bold_style = ParagraphStyle(
                name='TableTextBold', parent=table_text_normal_style, fontName=PDF_DEFAULT_FONT_BOLD)

            table_text_normal_centered_style = ParagraphStyle(
                name='TableTextNormalCentered', parent=table_text_normal_style, alignment=1)

            annotation_style = ParagraphStyle(
                name='AnnotationStyle', fontName='Helvetica-Oblique', fontSize=8, # Użycie bezpośrednio nazwy czcionki
                textColor=reportlab_colors.HexColor("#444444"), alignment=0,
                spaceBefore=0.4*cm)

            elements = []
            elements.append(Paragraph(self._transliterate("RAPORT ANALIZY PŁYTKI PCB"), title_style))
            elements.append(Paragraph(self._transliterate(f"Kod PCB: {self.current_pcb_code}"), subtitle_style))

            if not self.conn or not self.conn.is_connected():
                if hasattr(self, 'init_database_connection'):
                    self.init_database_connection() 
                else:
                    QMessageBox.critical(self, self._transliterate("Błąd Bazy Danych"), self._transliterate("Brak połączenia z bazą danych."))
                    return
            if not self.cursor:
                 QMessageBox.critical(self, self._transliterate("Błąd Bazy Danych"), self._transliterate("Kursor bazy danych nie jest dostępny."))
                 return
            
            self.cursor.execute("SELECT * FROM pcb_records WHERE pcb_code = %s", (self.current_pcb_code,))
            pcb_data = self.cursor.fetchone()
            
            self.cursor.execute("SELECT component_id, score FROM components WHERE pcb_code = %s ORDER BY score ASC, component_id", (self.current_pcb_code,))
            components_data = self.cursor.fetchall()

            if not pcb_data:
                QMessageBox.critical(self, self._transliterate("Błąd danych"), self._transliterate(f"Nie znaleziono danych rekordu dla PCB: {self.current_pcb_code}"))
                return

            elements.append(Paragraph(self._transliterate("INFORMACJE PODSTAWOWE"), section_heading_style))
            meta_data_list = [
                [Paragraph(self._transliterate("Kod PCB:"), table_text_bold_style), Paragraph(self._transliterate(str(self.current_pcb_code)), table_text_normal_style)],
                [Paragraph(self._transliterate("Data analizy:"), table_text_bold_style), Paragraph(self._transliterate(pcb_data['date_analyzed'].strftime("%Y-%m-%d %H:%M:%S")), table_text_normal_style)],
                [Paragraph(self._transliterate("Data generacji raportu:"), table_text_bold_style), Paragraph(self._transliterate(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), table_text_normal_style)]
            ]
            if self.user_data:
                user_name_surname = self._transliterate(f"{self.user_data.get('name', '')} {self.user_data.get('surname', '')}".strip())
                meta_data_list.append([Paragraph(self._transliterate("Operator:"), table_text_bold_style), Paragraph(user_name_surname, table_text_normal_style)])
                if self.user_data.get('position_name'):
                    meta_data_list.append([Paragraph(self._transliterate("Stanowisko:"), table_text_bold_style), Paragraph(self._transliterate(self.user_data['position_name']), table_text_normal_style)])
                if self.user_data.get('email'):
                    meta_data_list.append([Paragraph(self._transliterate("Email operatora:"), table_text_bold_style), Paragraph(self._transliterate(self.user_data['email']), table_text_normal_style)])
            
            meta_table = Table(meta_data_list, colWidths=[doc.width * 0.30, doc.width * 0.70])
            meta_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), 0.5, reportlab_colors.HexColor("#cccccc")),
                ('LEFTPADDING', (0,0),(-1,-1), 0), ('RIGHTPADDING', (0,0),(-1,-1), 0), 
                ('BOTTOMPADDING', (0,0),(-1,-1), 2), ('TOPPADDING', (0,0),(-1,-1), 2)
            ]))
            elements.append(meta_table)

            elements.append(Paragraph(self._transliterate("WYKRYTE KOMPONENTY"), section_heading_style))
            
            attention_required_for_note = False 

            if components_data:
                header_style_pdf_table = ParagraphStyle(name='HeaderCompTablePDF', parent=table_text_bold_style, alignment=0)
                header_style_pdf_table_centered = ParagraphStyle(name='HeaderCompTableCenteredPDF', parent=header_style_pdf_table, alignment=1)

                table_header_components_pdf = [
                    Paragraph(self._transliterate("ID Komponentu"), header_style_pdf_table),
                    Paragraph(self._transliterate("Wynik Oceny"), header_style_pdf_table_centered)
                ]
                
                styled_component_data = [table_header_components_pdf]
                
                dynamic_styles_for_table = [
                    ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.HexColor("#e0e0e0")),
                    ('GRID', (0, 0), (-1, -1), 0.5, reportlab_colors.HexColor("#bbbbbb")),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('LEFTPADDING', (0,0), (-1,-1), 5), ('RIGHTPADDING', (0,0), (-1,-1), 5),
                    ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3)
                ]

                for i, comp in enumerate(components_data):
                    row_index_in_table_style = i + 1
                    comp_id_raw = str(comp['component_id'])
                    score_val = comp['score']
                    score_text_pdf = f"{score_val:.2f}"
                    id_prefix_pdf = ""
                    current_id_text_style_pdf = table_text_normal_style
                    current_score_text_style_pdf = table_text_normal_centered_style

                    if score_val < 0.7:
                        attention_required_for_note = True
                        id_prefix_pdf = "(!) " 
                        dynamic_styles_for_table.append(
                            ('BACKGROUND', (0, row_index_in_table_style), (-1, row_index_in_table_style), reportlab_colors.orange)
                        )
                    
                    id_display_pdf = self._transliterate(id_prefix_pdf + comp_id_raw)
                    
                    styled_component_data.append([
                        Paragraph(id_display_pdf, current_id_text_style_pdf),
                        Paragraph(self._transliterate(score_text_pdf), current_score_text_style_pdf)
                    ])
                
                components_table_obj_pdf = Table(styled_component_data, colWidths=[doc.width * 0.65, doc.width * 0.35])
                components_table_obj_pdf.setStyle(TableStyle(dynamic_styles_for_table))
                elements.append(components_table_obj_pdf)
            else:
                elements.append(Paragraph(self._transliterate("Nie wykryto komponentów dla tej płytki PCB."), table_text_normal_style))
            
            if attention_required_for_note:
                elements.append(Spacer(1, 0.4 * cm))
                elements.append(Paragraph(self._transliterate("Uwaga: Elementy oznaczone kolorem pomarańczowym i/lub symbolem (!) mogą wymagać dodatkowej weryfikacji w celu zapewnienia dokładności analizy."), annotation_style))

            doc.build(elements)
            QMessageBox.information(self, self._transliterate("Sukces"), self._transliterate(f"Raport PDF został zapisany: {file_path}"))
        
        except mysql.connector.Error as e:
            QMessageBox.critical(self, self._transliterate("Błąd Bazy Danych (PDF)"), self._transliterate(f"Nie można wygenerować raportu (baza danych): {e}"))
            traceback.print_exc()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, self._transliterate("Błąd Generowania PDF"), self._transliterate(f"Wystąpił nieoczekiwany błąd: {e}"))

    def _transliterate(self, text):
        """mapowanie polskich znaków"""
        if not isinstance(text, str):
            return text
        
        mapping = {
            'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n',
            'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z',
            'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L', 'Ń': 'N',
            'Ó': 'O', 'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z',
        }
        for pl_char, lat_char in mapping.items():
            text = text.replace(pl_char, lat_char)
        return text

    def closeEvent(self, event):
        if hasattr(self, 'dateTimeTimer'): self.dateTimeTimer.stop()
        self._close_db_connection() 
        event.accept()

    def refresh_data(self):
        try:
            self.init_database_connection() 
            
            if not hasattr(self, 'reportCodeCombo') or not hasattr(self, 'dateEdit'): 
                print("BŁĄD: Nie można odświeżyć, brak reportCodeCombo lub dateEdit.")
                return

            previously_selected_pcb_code_in_combo = self.reportCodeCombo.currentData()
            self.load_report_codes() 

            self.dateEdit.setStyleSheet("")
            
            active_view_mode = getattr(self, 'current_view_mode', "date") 

            if active_view_mode == "all":
                self.load_all_pcb_data()
            elif active_view_mode == "components" and self.current_pcb_code:
                if self.reportCodeCombo.findData(self.current_pcb_code) != -1:
                    self.display_pcb_components(self.current_pcb_code)
                else: 
                    self.reportCodeCombo.setCurrentIndex(0) 
            else: 
                if self.reportCodeCombo.currentIndex() != 0 : 
                     self.reportCodeCombo.setCurrentIndex(0) 
                else: 
                    self.current_pcb_code = None
                    self.load_report_data() 
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Błąd Odświeżania", f"Wystąpił błąd podczas odświeżania danych: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    report_dialog = Reports() 
    report_dialog.show()
    sys.exit(app.exec_())