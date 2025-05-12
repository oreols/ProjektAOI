from PyQt5.QtWidgets import (QDialog, QTableWidgetItem, QDateEdit, QFileDialog, 
                          QMessageBox, QComboBox, QVBoxLayout, QLabel, QHBoxLayout,
                          QPushButton, QFrame, QHeaderView, QTextEdit, QTableWidget)
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QDate, pyqtSlot
from PyQt5.QtGui import QPixmap, QFont, QColor
import mysql.connector
from datetime import datetime, timedelta
import os
import sys
import traceback
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import inch, cm
from db_config import DB_CONFIG
import pkg_resources

# Inicjalizacja czcionek wbudowanych - nie próbujemy pobierać czcionek z sieci
try:
    # Używamy czcionek wbudowanych w ReportLab, które zawsze działają
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    
    # Upewniamy się, że pdfmetrics ma zarejestrowane podstawowe czcionki
    pdfmetrics.registerFont(pdfmetrics._fonts['Helvetica'])
    pdfmetrics.registerFont(pdfmetrics._fonts['Helvetica-Bold'])
    pdfmetrics.registerFont(pdfmetrics._fonts['Helvetica-Oblique'])
    pdfmetrics.registerFont(pdfmetrics._fonts['Helvetica-BoldOblique'])
    
    # Rejestrujemy rodzinę czcionek
    registerFontFamily('Helvetica', normal='Helvetica', bold='Helvetica-Bold',
                       italic='Helvetica-Oblique', boldItalic='Helvetica-BoldOblique')
    
    print("Podstawowe czcionki zarejestrowane pomyślnie")
except Exception as e:
    print(f"Błąd rejestracji czcionek: {e}")

class Reports(QDialog):
    def __init__(self, user_id=None):
        super(Reports, self).__init__()
        loadUi("ui/report.ui", self)
        
        self.user_id = user_id
        self.user_data = None
        
        try:
            # Initialize database connection
            self.init_database()
            
            # If user_id is provided, get user information
            if self.user_id:
                self.user_data = self.get_user_data()
            
            # Setup UI connections
            self.setup_ui()
            
            # Load initial report data (today's data)
            self.load_report_data()
            
        except Exception as e:
            print(f"Błąd inicjalizacji raportu: {e}")
            traceback.print_exc()
            QMessageBox.critical(None, "Błąd", f"Wystąpił błąd podczas inicjalizacji: {e}")

    def init_database(self):
        """Initialize database connection"""
        try:
            print("Łączenie z bazą danych...")
            self.conn = mysql.connector.connect(**DB_CONFIG)
            self.cursor = self.conn.cursor(dictionary=True)
            print("Połączono z bazą danych")
        except mysql.connector.Error as e:
            print(f"Błąd połączenia z bazą danych: {e}")
            QMessageBox.critical(self, "Błąd", f"Nie można połączyć się z bazą danych: {e}")
            raise e

    def get_user_data(self):
        """Get user information from database"""
        try:
            self.cursor.execute("""
                SELECT u.id, u.name, u.surname, u.email, p.name AS position_name 
                FROM user u 
                LEFT JOIN `position` p ON u.position_id = p.id 
                WHERE u.id = %s
            """, (self.user_id,))
            return self.cursor.fetchone()
        except mysql.connector.Error as e:
            print(f"Błąd pobierania danych użytkownika: {e}")
            return None

    def setup_ui(self):
        """Setup UI connections and initial state"""
        # Usuniete przyciski zakresu dat (dzien, tydzien, miesiac, rok)
        
        # Ustawienie jednolitego stylu dla całego interfejsu
        self.setStyleSheet("""
            QDialog {
                background-color: #2f3136;
                color: #e8e9ea;
            }
            QTableWidget {
                background-color: #36393f;
                border: none;
                gridline-color: #4f545c;
                color: #dcddde;
                font-size: 14px;
                selection-background-color: #5865F2;
                selection-color: #ffffff;
                outline: none;
            }
            QTableWidget::item {
                padding: 12px 10px;
                min-height: 30px;
            }
            QHeaderView::section {
                background-color: #2f3136;
                color: #ffffff;
                padding: 15px 12px;
                border: none;
                border-bottom: 1px solid #4f545c;
                min-height: 30px;
            }
            QHeaderView::section:horizontal {
                border-right: 1px solid #4f545c;
            }
            QHeaderView::section:vertical {
                background-color: #40444b;
                border-right: 1px solid #4f545c;
                border-bottom: 1px solid #4f545c;
            }
            QFrame {
                background-color: #36393f;
                border: none;
            }
            QLabel {
                color: #dcddde;
            }
            QComboBox, QDateEdit {
                background-color: #40444b;
                color: #dcddde;
                border: 1px solid #4f545c;
                border-radius: 4px;
                padding: 5px;
                min-height: 28px;
            }
            QComboBox:focus, QDateEdit:focus {
                border: 1px solid #5865F2;
            }
            QComboBox QAbstractItemView {
                background-color: #36393f;
                color: #dcddde;
                border: 1px solid #4f545c;
                selection-background-color: #5865F2;
            }
            QPushButton {
                background-color: #5865F2;
                color: white;
                font-size: 13px;
                font-weight: 600;
                padding: 6px 16px;
                border-radius: 4px;
                border: none;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #4e5dcf;
            }
            QPushButton:pressed {
                background-color: #404ab8;
            }
            QPushButton#refreshButton {
                background-color: #4f545c;
                color: white;
            }
            QPushButton#refreshButton:hover {
                background-color: #5d6269;
            }
            QPushButton#refreshButton:pressed {
                background-color: #6b6f79;
            }
        """)
        
        # Setup date selector if exists in UI
        if hasattr(self, 'dateEdit'):
            self.dateEdit.setDate(QDate.currentDate())
            self.dateEdit.dateChanged.connect(self.on_date_changed)
        else:
            # Create date selector if not in UI
            self.dateFrame = QFrame(self)
            date_layout = QHBoxLayout(self.dateFrame)
            date_layout.setContentsMargins(10, 10, 10, 10)
            date_label = QLabel("Wybierz datę:", self.dateFrame)
            self.dateEdit = QDateEdit(self.dateFrame)
            self.dateEdit.setCalendarPopup(True)
            self.dateEdit.setDate(QDate.currentDate())
            self.dateEdit.dateChanged.connect(self.on_date_changed)
            date_layout.addWidget(date_label)
            date_layout.addWidget(self.dateEdit)
            
            # Add to main layout after the top toolbar
            self.mainVerticalLayout.insertWidget(1, self.dateFrame)
        
        # Setup report code selector
        if not hasattr(self, 'reportCodeCombo'):
            self.reportCodeFrame = QFrame(self)
            code_layout = QHBoxLayout(self.reportCodeFrame)
            code_layout.setContentsMargins(10, 10, 10, 10)
            code_label = QLabel("Kod PCB:", self.reportCodeFrame)
            self.reportCodeCombo = QComboBox(self.reportCodeFrame)
            self.reportCodeCombo.setMinimumWidth(200)
            self.reportCodeCombo.currentIndexChanged.connect(self.on_report_code_changed)
            code_layout.addWidget(code_label)
            code_layout.addWidget(self.reportCodeCombo)
            
            # Add to main layout after the date selector
            self.mainVerticalLayout.insertWidget(2, self.reportCodeFrame)
        
        # Flaga do śledzenia aktualnie wybranego PCB
        self.current_pcb_code = None
        
        # Setup refresh button
        if hasattr(self, 'refreshButton'):
            self.refreshButton.clicked.connect(self.refresh_data)
            
        # Setup generate PDF button
        if not hasattr(self, 'generatePdfButton'):
            self.generatePdfButton = QPushButton("Generuj raport PDF", self)
            self.generatePdfButton.clicked.connect(self.generate_pdf_report)
            
            # Create a horizontal layout for the bottom bar buttons
            if hasattr(self, 'bottomBar'):
                layout = self.bottomBar.layout()
                if layout:
                    # Add the generate PDF button to the existing layout
                    layout.addWidget(self.generatePdfButton)
                else:
                    # Create a new layout for the bottom bar
                    bottom_layout = QHBoxLayout(self.bottomBar)
                    bottom_layout.addStretch()
                    bottom_layout.addWidget(self.generatePdfButton)
            else:
                # Just add it to the main layout if there's no bottom bar
                self.mainVerticalLayout.addWidget(self.generatePdfButton)
        
        # Popraw ustawienia tabeli, aby zapewnić lepszą czytelność
        if hasattr(self, 'tableWidget'):
            self.tableWidget.setAlternatingRowColors(True)
            self.tableWidget.setShowGrid(True)
            self.tableWidget.setWordWrap(True)
            self.tableWidget.verticalHeader().setDefaultSectionSize(50)  # Zwiększona wysokość wiersza
            self.tableWidget.horizontalHeader().setDefaultSectionSize(180)  # Zwiększona szerokość kolumny
            
            # Włącz scroll per pixel dla płynniejszego przewijania
            self.tableWidget.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
            self.tableWidget.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        
        # Set title
        if hasattr(self, 'titleLabel'):
            self.titleLabel.setText("Raporty analizy PCB")
        
        # Load available report codes
        self.load_report_codes()

    def load_report_codes(self):
        """Load all available PCB codes for report selection"""
        try:
            self.cursor.execute("""
                SELECT pcb_code, date_analyzed 
                FROM pcb_records 
                ORDER BY date_analyzed DESC
            """)
            records = self.cursor.fetchall()
            
            if hasattr(self, 'reportCodeCombo'):
                self.reportCodeCombo.clear()
                self.reportCodeCombo.addItem("-- Wybierz kod PCB --", None)
                
                for record in records:
                    display_text = f"{record['pcb_code']} ({record['date_analyzed'].strftime('%Y-%m-%d')})"
                    self.reportCodeCombo.addItem(display_text, record['pcb_code'])
        
        except mysql.connector.Error as e:
            print(f"Błąd ładowania kodów raportów: {e}")
            QMessageBox.warning(self, "Ostrzeżenie", f"Nie można załadować kodów PCB: {e}")

    def on_date_changed(self):
        """Handle date change event"""
        self.load_report_data()

    def on_report_code_changed(self, index):
        """Handle report code selection change"""
        if index <= 0:  # First item is placeholder
            return
        
        pcb_code = self.reportCodeCombo.itemData(index)
        self.current_pcb_code = pcb_code  # Zapisz aktualnie wybrany PCB
        
        # Najpierw załaduj dane o PCB
        self.load_specific_report(pcb_code)
        
        # Następnie pobierz i wyświetl komponenty dla wybranego PCB
        self.display_pcb_components(pcb_code)

    def display_pcb_components(self, pcb_code):
        """Wyświetl komponenty dla wybranego PCB w formie listy lub zgrupowane"""
        try:
            # Pobierz komponenty dla tego PCB
            self.cursor.execute("""
                SELECT component_id, component_type, score, bbox 
                FROM components 
                WHERE pcb_code = %s
                ORDER BY component_type, component_id
            """, (pcb_code,))
            
            components = self.cursor.fetchall()
            
            if not components:
                QMessageBox.information(self, "Informacja", f"Nie znaleziono komponentów dla PCB: {pcb_code}")
                return
            
            # Wyczyść istniejącą tabelę
            self.tableWidget.clear()
            self.tableWidget.setRowCount(0)
            
            # Widok prostej listy komponentów
            self.display_component_list(components)
            
            # Ustaw tytuł z informacją o PCB
            if hasattr(self, 'titleLabel'):
                count_text = f"{len(components)} komponentów"
                self.titleLabel.setText(f"PCB: {pcb_code} ({count_text})")
            
        except mysql.connector.Error as e:
            print(f"Błąd pobierania komponentów: {e}")
            QMessageBox.warning(self, "Ostrzeżenie", f"Nie można pobrać komponentów: {e}")
    
    def display_component_list(self, components):
        """Wyświetl komponenty jako prostą listę"""
        # Sprawdź czy lista komponentów jest pusta
        if not components:
            self.tableWidget.clear()
            self.tableWidget.setRowCount(0)
            self.tableWidget.setColumnCount(1)
            self.tableWidget.setHorizontalHeaderLabels(['Informacja'])
            self.tableWidget.insertRow(0)
            info_item = QTableWidgetItem("Brak danych do wyświetlenia")
            info_item.setTextAlignment(Qt.AlignCenter)
            self.tableWidget.setItem(0, 0, info_item)
            return
            
        # Ustaw kolumny dla widoku listy
        self.tableWidget.clear()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(['ID komponentu', 'Wynik'])
        
        # Ustaw szerokości kolumn
        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # ID komponentu
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Wynik
        
        # Zwiększ domyślną wysokość wierszy dla lepszej czytelności
        self.tableWidget.verticalHeader().setDefaultSectionSize(50)
        
        # Dodaj komponenty do tabeli
        for i, comp in enumerate(components):
            self.tableWidget.insertRow(i)
            
            # ID komponentu
            id_item = QTableWidgetItem(comp['component_id'])
            id_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.tableWidget.setItem(i, 0, id_item)
            
            # Wynik
            score_item = QTableWidgetItem(f"{comp['score']:.2f}")
            score_item.setTextAlignment(Qt.AlignCenter)
            self.tableWidget.setItem(i, 1, score_item)
            
            # Kolorowanie wiersza w zależności od wyniku
            if comp['score'] < 0.5:
                # Ciemny czerwony dla lepszego kontrastu z ciemnym tłem
                bg_color = QColor(165, 30, 30)
                for col in range(2):
                    self.tableWidget.item(i, col).setBackground(bg_color)
                    self.tableWidget.item(i, col).setForeground(Qt.white)
            elif comp['score'] < 0.8:
                # Ciemny żółty dla lepszego kontrastu
                bg_color = QColor(150, 135, 0)
                for col in range(2):
                    self.tableWidget.item(i, col).setBackground(bg_color)
                    self.tableWidget.item(i, col).setForeground(Qt.black)
            else:
                # Dla wysokich wyników dodaj delikatne tło, by zaznaczyć dobry wynik
                bg_color = QColor(30, 130, 76, 120)  # Ciemny zielony z przezroczystością
                for col in range(2):
                    self.tableWidget.item(i, col).setBackground(bg_color)
                    # Jasny text dla lepszej czytelności
                    self.tableWidget.item(i, col).setForeground(Qt.white)
                    
        # Dopasowujemy szerokość kolumn do zawartości, ale z minimalną szerokością
        self.tableWidget.resizeColumnsToContents()
        min_width = 180
        if self.tableWidget.columnWidth(0) < min_width:
            self.tableWidget.setColumnWidth(0, min_width)
        if self.tableWidget.columnWidth(1) < 100:
            self.tableWidget.setColumnWidth(1, 100)
    
    def load_report_data(self):
        """Load report data based on selected date"""
        try:
            selected_date = self.dateEdit.date().toPyDate()
            
            # Format for SQL query
            date_str = selected_date.strftime('%Y-%m-%d')
            
            self.cursor.execute("""
                SELECT p.pcb_code, p.date_analyzed, COUNT(c.id) as component_count
                FROM pcb_records p
                LEFT JOIN components c ON p.pcb_code = c.pcb_code
                WHERE DATE(p.date_analyzed) = %s
                GROUP BY p.pcb_code, p.date_analyzed
                ORDER BY p.date_analyzed DESC
            """, (date_str,))
            
            records = self.cursor.fetchall()
            self.display_report_data(records)
            
        except mysql.connector.Error as e:
            print(f"Błąd ładowania danych raportu: {e}")
            QMessageBox.warning(self, "Ostrzeżenie", f"Nie można załadować danych raportu: {e}")

    def load_specific_report(self, pcb_code):
        """Load report data for a specific PCB code"""
        try:
            self.cursor.execute("""
                SELECT p.pcb_code, p.date_analyzed, COUNT(c.id) as component_count
                FROM pcb_records p
                LEFT JOIN components c ON p.pcb_code = c.pcb_code
                WHERE p.pcb_code = %s
                GROUP BY p.pcb_code, p.date_analyzed
            """, (pcb_code,))
            
            records = self.cursor.fetchall()
            self.display_report_data(records)
            
        except mysql.connector.Error as e:
            print(f"Błąd ładowania konkretnego raportu: {e}")
            QMessageBox.warning(self, "Ostrzeżenie", f"Nie można załadować raportu dla kodu PCB {pcb_code}: {e}")

    def display_report_data(self, records):
        """Display report data in the table"""
        # Clear existing table
        self.tableWidget.clear()
        self.tableWidget.setRowCount(0)
        
        if not records:
            # Wyświetl informację o braku danych
            self.tableWidget.setColumnCount(1)
            self.tableWidget.setHorizontalHeaderLabels(['Informacja'])
            self.tableWidget.insertRow(0)
            info_item = QTableWidgetItem("Brak danych dla wybranego okresu")
            info_item.setTextAlignment(Qt.AlignCenter)
            self.tableWidget.setItem(0, 0, info_item)
            return
        
        # Set up table headers if not already set
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setHorizontalHeaderLabels(['Kod PCB', 'Data analizy', 'Liczba komponentów'])
        
        # Set column widths
        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        # Add data to table
        for row_idx, record in enumerate(records):
            self.tableWidget.insertRow(row_idx)
            
            # PCB Code
            pcb_code_item = QTableWidgetItem(record['pcb_code'])
            pcb_code_item.setData(Qt.UserRole, record['pcb_code'])
            pcb_code_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.tableWidget.setItem(row_idx, 0, pcb_code_item)
            
            # Analysis Date
            date_str = record['date_analyzed'].strftime('%Y-%m-%d %H:%M')
            date_item = QTableWidgetItem(date_str)
            date_item.setTextAlignment(Qt.AlignCenter)
            self.tableWidget.setItem(row_idx, 1, date_item)
            
            # Component Count
            count_item = QTableWidgetItem(str(record['component_count']))
            count_item.setTextAlignment(Qt.AlignCenter)
            self.tableWidget.setItem(row_idx, 2, count_item)
        
        # Dopasuj szerokości kolumn do zawartości, ale z minimalnymi szerokościami
        self.tableWidget.resizeColumnsToContents()
        min_widths = [180, 150, 100]  # Minimalne szerokości dla poszczególnych kolumn
        for col, min_width in enumerate(min_widths):
            if self.tableWidget.columnWidth(col) < min_width:
                self.tableWidget.setColumnWidth(col, min_width)
        
        # Setup item double-click to view component details
        self.tableWidget.itemDoubleClicked.connect(self.show_component_details)

    def show_component_details(self, item):
        """Show detailed component information for the selected PCB"""
        row = item.row()
        pcb_code = self.tableWidget.item(row, 0).data(Qt.UserRole)
        
        try:
            # Get components for this PCB
            self.cursor.execute("""
                SELECT component_id, component_type, score, bbox 
                FROM components 
                WHERE pcb_code = %s
                ORDER BY component_type, component_id
            """, (pcb_code,))
            
            components = self.cursor.fetchall()
            
            if not components:
                QMessageBox.information(self, "Szczegóły komponentów", f"Nie znaleziono komponentów dla PCB: {pcb_code}")
                return
            
            # Get PCB analysis date
            self.cursor.execute("""
                SELECT date_analyzed 
                FROM pcb_records 
                WHERE pcb_code = %s
            """, (pcb_code,))
            
            pcb_record = self.cursor.fetchone()
            analysis_date = pcb_record['date_analyzed'] if pcb_record else "Nieznana"
            
            # Prepare detailed message
            details = f"PCB: {pcb_code}\n"
            details += f"Data analizy: {analysis_date}\n"
            details += f"Liczba komponentów: {len(components)}\n\n"
            
            # Group components by type
            components_by_type = {}
            for comp in components:
                comp_type = comp['component_type'] or "Nieznany"
                if comp_type not in components_by_type:
                    components_by_type[comp_type] = []
                
                components_by_type[comp_type].append(comp)
            
            # Display components grouped by type
            details += "Wykryte komponenty:\n"
            for component_type, comps in components_by_type.items():
                details += f"\n== {component_type} ({len(comps)}) ==\n"
                for comp in comps:
                    details += f"- {comp['component_id']} (Wynik: {comp['score']:.2f})\n"
            
            # Show details
            QMessageBox.information(self, "Szczegóły komponentów", details)
            
        except mysql.connector.Error as e:
            print(f"Błąd pobierania szczegółów komponentów: {e}")
            QMessageBox.warning(self, "Ostrzeżenie", f"Nie można pobrać szczegółów komponentów: {e}")

    def generate_pdf_report(self):
        """Generate a PDF report for the currently viewed data"""
        if not self.current_pcb_code:
            QMessageBox.warning(self, "Ostrzeżenie", "Wybierz najpierw kod PCB do wygenerowania raportu")
            return
        
        # Ask for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Zapisz raport PDF", f"Raport_PCB_{self.current_pcb_code}.pdf", "Pliki PDF (*.pdf)"
        )
        
        if not file_path:
            return  # User canceled
        
        if not file_path.lower().endswith('.pdf'):
            file_path += '.pdf'
        
        try:
            # Create the PDF document
            doc = SimpleDocTemplate(
                file_path,
                pagesize=A4,
                topMargin=1*cm,
                bottomMargin=1*cm,
                leftMargin=2*cm,
                rightMargin=2*cm
            )
            
            # Używamy standardowych czcionek wbudowanych w ReportLab
            standard_font = 'Helvetica'
            bold_font = 'Helvetica-Bold'
            italic_font = 'Helvetica-Oblique'
            
            # Pobieramy podstawowe style
            styles = getSampleStyleSheet()
            
            # Tworzymy własne style
            title_style = ParagraphStyle(
                name='CustomTitle',
                fontName=bold_font,
                fontSize=16,
                alignment=1,  # Center
                spaceAfter=0.5*cm
            )
            
            subtitle_style = ParagraphStyle(
                name='CustomSubtitle',
                fontName=italic_font,
                fontSize=12,
                alignment=1,  # Center
                spaceAfter=0.3*cm
            )
            
            heading_style = ParagraphStyle(
                name='CustomHeading',
                fontName=bold_font,
                fontSize=12,
                spaceBefore=0.4*cm,
                spaceAfter=0.2*cm
            )
            
            normal_style = ParagraphStyle(
                name='CustomNormal',
                fontName=standard_font,
                fontSize=10,
                spaceBefore=0.1*cm,
                spaceAfter=0.1*cm
            )
            
            # Elementy raportu
            elements = []
            
            # Nagłówek raportu
            elements.append(Paragraph("PCB ANALYSIS REPORT", title_style))
            elements.append(Paragraph(f"PCB Code: {self.current_pcb_code}", subtitle_style))
            elements.append(Spacer(1, 0.5*cm))
            
            # Pobierz dane PCB z bazy danych
            try:
                # Dane PCB
                self.cursor.execute("""
                    SELECT * FROM pcb_records 
                    WHERE pcb_code = %s
                """, (self.current_pcb_code,))
                pcb_data = self.cursor.fetchone()
                
                # Dane komponentów
                self.cursor.execute("""
                    SELECT component_id, component_type, score, bbox 
                    FROM components 
                    WHERE pcb_code = %s
                    ORDER BY component_type, score DESC
                """, (self.current_pcb_code,))
                components = self.cursor.fetchall()
                
                # Jeśli nie ma danych, wyświetl błąd
                if not pcb_data or not components:
                    QMessageBox.warning(self, "Brak danych", f"Nie znaleziono danych dla PCB o kodzie {self.current_pcb_code}")
                    return
                
                # ===== SEKCJA 1: METADANE =====
                elements.append(Paragraph("REPORT INFORMATION", heading_style))
                
                # Tabela metadanych
                metadata = [
                    ["PCB Code:", self.current_pcb_code],
                    ["Analysis Date:", pcb_data['date_analyzed'].strftime("%Y-%m-%d %H:%M:%S")],
                    ["Report Generation Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                ]
                
                # Dodaj informacje o operatorze, jeśli dostępne
                if self.user_data:
                    metadata.append(["Operator:", f"{self.user_data['name']} {self.user_data['surname']}"])
                    if self.user_data['position_name']:
                        metadata.append(["Position:", self.user_data['position_name']])
                    if self.user_data['email']:
                        metadata.append(["Email:", self.user_data['email']])
                
                # Tabela metadanych
                meta_table = Table(metadata, colWidths=[doc.width*0.3, doc.width*0.7])
                meta_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (0, -1), bold_font),
                    ('FONTNAME', (1, 0), (1, -1), standard_font),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                    ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                
                elements.append(meta_table)
                elements.append(Spacer(1, 0.5*cm))
                
                # ===== SEKCJA 3: SZCZEGÓŁOWA LISTA KOMPONENTÓW =====
                elements.append(Paragraph("DETAILS COMPONENTS LIST", heading_style))
                
                # Tworzymy tabelę z wszystkimi komponentami
                all_components_data = [["ID", "Wynik"]]
                
                # Dodajemy wszystkie komponenty do tabeli
                for comp in components:
                    all_components_data.append([
                        comp['component_id'],
                        f"{comp['score']:.2f}"
                    ])
                
                # Tworzymy stylowaną tabelę komponentów
                components_table = Table(all_components_data, colWidths=[doc.width*0.7, doc.width*0.3])
                
                # Podstawowy styl tabeli
                table_style = [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),  # Nagłówek
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                    ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), bold_font),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                ]
                
                # Ustawienie stylu tabeli
                components_table.setStyle(TableStyle(table_style))
                
                elements.append(components_table)
                
                # Budowanie PDF
                doc.build(elements)
                
                QMessageBox.information(self, "Sukces", f"Raport zapisany do {file_path}")
                
            except mysql.connector.Error as e:
                print(f"Błąd bazy danych podczas generowania raportu: {e}")
                QMessageBox.critical(self, "Błąd", f"Nie można wygenerować raportu: {e}")
            except Exception as e:
                print(f"Błąd podczas generowania raportu PDF: {e}")
                traceback.print_exc()
                QMessageBox.critical(self, "Błąd", f"Nie można wygenerować raportu PDF: {e}")
        
        except Exception as e:
            print(f"Błąd tworzenia dokumentu PDF: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Błąd", f"Nie można utworzyć dokumentu PDF: {e}")

    def add_detailed_component_analysis(self, pcb_code, elements, styles, doc):
        """Add detailed component analysis for a specific PCB"""
        # Ta metoda nie jest już używana, nowa implementacja jest w generate_pdf_report
        pass

    def closeEvent(self, event):
        """Handle window close event"""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.cursor.close()
                self.conn.close()
        except:
            pass
        event.accept()

    def refresh_data(self):
        """Odświeża dane w tabeli na podstawie bieżących ustawień"""
        if self.current_pcb_code:
            # Jeśli wybrany jest konkretny kod PCB, odśwież jego dane
            self.load_specific_report(self.current_pcb_code)
            # Odśwież również dane komponentów
            self.display_pcb_components(self.current_pcb_code)
        else:
            # W przeciwnym razie odśwież wszystkie dane
            self.load_report_data()
        
        # Odśwież listę dostępnych kodów PCB
        self.load_report_codes()
        
        QMessageBox.information(self, "Odświeżanie", "Dane zostały odświeżone pomyślnie.")
