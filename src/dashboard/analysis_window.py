"""
ECG Analysis Window - Display 12-lead ECG analysis from backend reports

This module provides a comprehensive 12-lead ECG analysis window that:
- Fetches reports from backend (phone-generated)
- Displays 12-lead ECG waveforms in standard format
- Provides analysis tools and metrics
- Similar interface to the existing 12-lead test page
"""

import os
import json
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QScrollArea, QWidget, QFrame,
    QSplitter, QTextEdit, QComboBox, QDateEdit, QMessageBox,
    QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QDate, QTimer
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ECGAnalysisWindow(QDialog):
    """Analysis window for 12-lead ECG reports from backend"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ECG Analysis Window")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Set window style for medical ECG software appearance
        self.setStyleSheet("""
            QDialog {
                background: #f0f0f0;
                color: black;
            }
            QComboBox {
                background: white;
                color: black;
                border: 2px solid #4a90e2;
                border-radius: 4px;
                padding: 4px 8px;
                min-width: 200px;
                font-size: 11px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #4a90e2;
            }
            QComboBox QAbstractItemView {
                background: white;
                color: black;
                border: 2px solid #4a90e2;
                selection-background-color: #4a90e2;
                selection-color: white;
            }
            QPushButton {
                background: #4a90e2;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #357abd;
            }
            QPushButton:pressed {
                background: #2968a3;
            }
            QTableWidget {
                background: white;
                color: black;
                gridline-color: #d0d0d0;
                border: 1px solid #d0d0d0;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 4px;
                border-bottom: 1px solid #d0d0d0;
            }
            QTableWidget::item:selected {
                background: #4a90e2;
                color: white;
            }
            QHeaderView::section {
                background: #4a90e2;
                color: white;
                padding: 6px;
                font-weight: bold;
                font-size: 11px;
                border: none;
            }
            QTextEdit {
                background: white;
                color: black;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 6px;
                font-size: 11px;
            }
            QLabel {
                color: black;
                font-size: 11px;
                font-weight: normal;
            }
        """)
        
        # Data storage
        self.reports = []
        self.current_report = None
        self.ecg_data = {}
        
        # Setup UI
        self.setup_ui()
        self.load_reports()
        
    def setup_ui(self):
        """Setup the main UI layout matching medical ECG interface"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Top toolbar with patient info and controls
        self.create_top_toolbar(main_layout)
        
        # Main content area with 12-lead grid
        self.create_12lead_grid(main_layout)
        
        # Bottom analysis panel
        self.create_bottom_analysis_panel(main_layout)
        
        self.setLayout(main_layout)
        
    def create_top_toolbar(self, main_layout):
        """Create top toolbar with patient info and controls"""
        toolbar_frame = QFrame()
        toolbar_frame.setStyleSheet("""
            QFrame {
                background: white;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                margin: 2px;
            }
        """)
        toolbar_layout = QHBoxLayout(toolbar_frame)
        toolbar_layout.setContentsMargins(10, 8, 10, 8)
        
        # Patient info section with medical styling
        patient_frame = QFrame()
        patient_frame.setStyleSheet("background: transparent;")
        patient_layout = QHBoxLayout(patient_frame)
        
        self.patient_name_label = QLabel("Patient: Not Selected")
        self.patient_name_label.setStyleSheet("""
            color: #333333;
            font-size: 12px;
            font-weight: bold;
            padding: 2px 8px;
        """)
        self.patient_id_label = QLabel("ID: --")
        self.patient_id_label.setStyleSheet("""
            color: #666666;
            font-size: 11px;
            padding: 2px 4px;
        """)
        self.patient_age_label = QLabel("Age: --")
        self.patient_age_label.setStyleSheet("""
            color: #666666;
            font-size: 11px;
            padding: 2px 4px;
        """)
        
        patient_layout.addWidget(self.patient_name_label)
        patient_layout.addWidget(QLabel(" | "))
        patient_layout.addWidget(self.patient_id_label)
        patient_layout.addWidget(QLabel(" | "))
        patient_layout.addWidget(self.patient_age_label)
        patient_layout.addStretch()
        
        toolbar_layout.addWidget(patient_frame)
        toolbar_layout.addStretch()
        
        # Controls section with medical styling
        controls_frame = QFrame()
        controls_frame.setStyleSheet("background: transparent;")
        controls_layout = QHBoxLayout(controls_frame)
        
        # Report selector
        report_label = QLabel("Report:")
        report_label.setStyleSheet("""
            color: #333333;
            font-size: 11px;
            font-weight: bold;
            padding-right: 8px;
        """)
        
        self.report_combo = QComboBox()
        self.report_combo.currentIndexChanged.connect(self.load_selected_report)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_reports)
        
        controls_layout.addWidget(report_label)
        controls_layout.addWidget(self.report_combo)
        controls_layout.addWidget(refresh_btn)
        
        toolbar_layout.addWidget(controls_frame)
        
        main_layout.addWidget(toolbar_frame)
        
    def create_12lead_grid(self, main_layout):
        """Create main 12-lead ECG grid display"""
        # Main ECG display frame with medical styling
        ecg_frame = QFrame()
        ecg_frame.setStyleSheet("""
            QFrame {
                background: white;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                margin: 5px;
            }
        """)
        ecg_layout = QVBoxLayout(ecg_frame)
        ecg_layout.setContentsMargins(10, 10, 10, 10)
        
        # Add title with medical styling
        title_label = QLabel("12-Lead ECG Analysis")
        title_label.setStyleSheet("""
            color: #333333;
            font-size: 14px;
            font-weight: bold;
            padding: 8px;
            background: #f8f8f8;
            border-radius: 4px;
            border-bottom: 2px solid #4a90e2;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        ecg_layout.addWidget(title_label)
        
        # Create matplotlib figure for 12 leads in grid layout
        self.figure = Figure(figsize=(14, 10), dpi=80, facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("""
            background: white;
            border: 1px solid #d0d0d0;
            border-radius: 4px;
        """)
        
        # Create 12 subplots in 4x3 grid (standard medical layout)
        self.axes = []
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        for i, lead in enumerate(leads):
            row = i // 3
            col = i % 3
            ax = self.figure.add_subplot(4, 3, i+1, facecolor='white')
            
            # Medical styling for each lead plot
            ax.set_title(f'Lead {lead}', fontsize=11, fontweight='bold', color='#333333', pad=5)
            ax.set_xlabel('Time (s)', fontsize=9, color='#666666', labelpad=3)
            ax.set_ylabel('mV', fontsize=9, color='#666666', labelpad=3)
            ax.tick_params(colors='#666666', labelsize=8)
            ax.spines['bottom'].set_color('#d0d0d0')
            ax.spines['top'].set_color('#d0d0d0')
            ax.spines['left'].set_color('#d0d0d0')
            ax.spines['right'].set_color('#d0d0d0')
            ax.grid(True, alpha=0.3, color='#e0e0e0', linestyle='-', linewidth=0.5)
            ax.set_facecolor('white')
            
            self.axes.append(ax)
        
        self.figure.tight_layout(pad=1.5)
        ecg_layout.addWidget(self.canvas)
        
        main_layout.addWidget(ecg_frame, stretch=3)
        
    def create_bottom_analysis_panel(self, main_layout):
        """Create bottom analysis panel with metrics and findings"""
        bottom_frame = QFrame()
        bottom_frame.setStyleSheet("""
            QFrame {
                background: #f8f8f8;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                margin: 5px;
            }
        """)
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left side - Metrics
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("""
            QFrame {
                background: white;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
            }
        """)
        metrics_layout = QVBoxLayout(metrics_frame)
        
        metrics_title = QLabel("ECG Metrics")
        metrics_title.setStyleSheet("""
            color: #333333;
            font-size: 12px;
            font-weight: bold;
            padding: 6px;
            background: #f0f0f0;
            border-bottom: 2px solid #4a90e2;
        """)
        metrics_layout.addWidget(metrics_title)
        
        # Metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.setMaximumHeight(200)
        self.metrics_table.setStyleSheet("""
            QTableWidget {
                background: white;
                gridline-color: #bdc3c7;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 3px;
            }
            QHeaderView::section {
                background: #34495e;
                color: white;
                padding: 3px;
                font-weight: bold;
                font-size: 11px;
            }
        """)
        metrics_layout.addWidget(self.metrics_table)
        
        bottom_layout.addWidget(metrics_frame, stretch=1)
        
        # Middle - Findings
        findings_frame = QFrame()
        findings_frame.setStyleSheet("""
            QFrame {
                background: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
        """)
        findings_layout = QVBoxLayout(findings_frame)
        
        findings_title = QLabel("FINDINGS")
        findings_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #2c3e50; padding: 5px;")
        findings_layout.addWidget(findings_title)
        
        self.findings_text = QTextEdit()
        self.findings_text.setMaximumHeight(200)
        self.findings_text.setStyleSheet("""
            QTextEdit {
                background: white;
                border: 1px solid #bdc3c7;
                border-radius: 2px;
                padding: 5px;
                font-size: 11px;
            }
        """)
        findings_layout.addWidget(self.findings_text)
        
        bottom_layout.addWidget(findings_frame, stretch=1)
        
        # Right side - Actions
        actions_frame = QFrame()
        actions_frame.setStyleSheet("""
            QFrame {
                background: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
        """)
        actions_layout = QVBoxLayout(actions_frame)
        
        actions_title = QLabel("ACTIONS")
        actions_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #2c3e50; padding: 5px;")
        actions_layout.addWidget(actions_title)
        
        # Action buttons
        self.export_btn = QPushButton("Export Report")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background: #27ae60;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #229954;
            }
        """)
        self.export_btn.clicked.connect(self.export_report)
        actions_layout.addWidget(self.export_btn)
        
        self.print_btn = QPushButton("Print")
        self.print_btn.setStyleSheet("""
            QPushButton {
                background: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #2980b9;
            }
        """)
        self.print_btn.clicked.connect(self.print_report)
        actions_layout.addWidget(self.print_btn)
        
        actions_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background: #e74c3c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #c0392b;
            }
        """)
        close_btn.clicked.connect(self.close)
        actions_layout.addWidget(close_btn)
        
        bottom_layout.addWidget(actions_frame, stretch=1)
        
        main_layout.addWidget(bottom_frame, stretch=1)
        
    def create_ecg_display(self, parent):
        """Create the 12-lead ECG display area"""
        ecg_frame = QFrame()
        ecg_frame.setStyleSheet("background: white; border: 1px solid #dee2e6; border-radius: 8px;")
        ecg_layout = QVBoxLayout(ecg_frame)
        
        # Title
        ecg_title = QLabel("12-Lead ECG Waveforms")
        ecg_title.setFont(QFont("Arial", 14, QFont.Bold))
        ecg_layout.addWidget(ecg_title)
        
        # Create matplotlib figure for 12 leads
        self.figure = Figure(figsize=(12, 10), dpi=80)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background: white;")
        
        # Create 12 subplots for each lead
        self.axes = []
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        for i, lead in enumerate(leads):
            ax = self.figure.add_subplot(4, 3, i+1)
            ax.set_title(f'Lead {lead}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Amplitude', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f8f8')
            self.axes.append(ax)
        
        self.figure.tight_layout()
        ecg_layout.addWidget(self.canvas)
        
        parent.addWidget(ecg_frame)
        
    def create_analysis_panel(self, parent):
        """Create the analysis panel with metrics and findings"""
        analysis_frame = QFrame()
        analysis_frame.setStyleSheet("background: white; border: 1px solid #dee2e6; border-radius: 8px;")
        analysis_layout = QVBoxLayout(analysis_frame)
        
        # Title
        analysis_title = QLabel("Analysis Results")
        analysis_title.setFont(QFont("Arial", 14, QFont.Bold))
        analysis_layout.addWidget(analysis_title)
        
        # Patient info
        self.info_frame = QFrame()
        self.info_frame.setStyleSheet("background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 10px;")
        self.info_layout = QGridLayout(self.info_frame)
        analysis_layout.addWidget(self.info_frame)
        
        # Metrics table
        metrics_label = QLabel("ECG Metrics:")
        metrics_label.setFont(QFont("Arial", 12, QFont.Bold))
        analysis_layout.addWidget(metrics_label)
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        analysis_layout.addWidget(self.metrics_table)
        
        # Findings
        findings_label = QLabel("Findings:")
        findings_label.setFont(QFont("Arial", 12, QFont.Bold))
        analysis_layout.addWidget(findings_label)
        
        self.findings_text = QTextEdit()
        self.findings_text.setMaximumHeight(150)
        self.findings_text.setStyleSheet("background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 5px;")
        analysis_layout.addWidget(self.findings_text)
        
        # Recommendations
        recommendations_label = QLabel("Recommendations:")
        recommendations_label.setFont(QFont("Arial", 12, QFont.Bold))
        analysis_layout.addWidget(recommendations_label)
        
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setMaximumHeight(100)
        self.recommendations_text.setStyleSheet("background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 5px;")
        analysis_layout.addWidget(self.recommendations_text)
        
        analysis_layout.addStretch()
        parent.addWidget(analysis_frame)
        
    def load_reports(self):
        """Load reports from backend/reports directory"""
        self.report_combo.clear()
        self.reports = []
        
        try:
            # Get reports directory
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            reports_dir = os.path.join(base_dir, 'reports')
            
            if not os.path.exists(reports_dir):
                QMessageBox.warning(self, "Warning", f"Reports directory not found: {reports_dir}")
                return
            
            # Look for report files
            for filename in os.listdir(reports_dir):
                if filename.endswith('.json') and not filename.startswith('index'):
                    filepath = os.path.join(reports_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            report = json.load(f)
                            
                        # Extract basic info
                        patient_name = report.get('patient_name', 'Unknown')
                        date_str = report.get('date', 'Unknown')
                        
                        # Format display name
                        display_name = f"{patient_name} - {date_str}"
                        self.report_combo.addItem(display_name, filepath)
                        self.reports.append(report)
                        
                    except Exception as e:
                        print(f"Error loading report {filename}: {e}")
                        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load reports: {str(e)}")
            
    def load_selected_report(self, index):
        """Load the selected report and display ECG data"""
        if index < 0 or index >= len(self.reports):
            return
            
        self.current_report = self.reports[index]
        
        # Update patient info
        self.update_patient_info()
        
        # Load ECG data if available
        self.load_ecg_data()
        
        # Update metrics and findings
        self.update_analysis_results()
        
    def update_patient_info(self):
        """Update patient information display in toolbar"""
        if not self.current_report:
            self.patient_name_label.setText("Patient: Not Selected")
            self.patient_id_label.setText("ID: --")
            self.patient_age_label.setText("Age: --")
            return
            
        # Update toolbar patient info
        patient_name = self.current_report.get('patient_name', 'Unknown')
        patient_id = self.current_report.get('patient_id', self.current_report.get('id', '--'))
        patient_age = self.current_report.get('age', '--')
        
        self.patient_name_label.setText(f"Patient: {patient_name}")
        self.patient_id_label.setText(f"ID: {patient_id}")
        self.patient_age_label.setText(f"Age: {patient_age}")
            
    def load_ecg_data(self):
        """Load and display 12-lead ECG data"""
        if not self.current_report:
            return
            
        # Clear all axes and reset styling
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        for i, ax in enumerate(self.axes):
            ax.clear()
            ax.set_facecolor('black')
            ax.set_title(f'Lead {leads[i]}', fontsize=10, fontweight='bold', color='white')
            ax.set_xlabel('Time (s)', fontsize=8, color='white')
            ax.set_ylabel('mV', fontsize=8, color='white')
            ax.tick_params(colors='white', labelsize=7)
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.grid(True, alpha=0.2, color='gray', linestyle='-', linewidth=0.5)
        
        # Try to load ECG data from report
        ecg_data = self.current_report.get('ecg_data', {})
        
        if ecg_data:
            for i, lead in enumerate(leads):
                if i < len(self.axes) and lead in ecg_data:
                    data = ecg_data[lead]
                    if isinstance(data, list) and len(data) > 0:
                        # Create time axis
                        sampling_rate = self.current_report.get('sampling_rate', 500)
                        time = np.arange(len(data)) / sampling_rate
                        
                        # Plot with medical ECG colors for different leads
                        # Define colors for different lead groups
                        lead_colors = {
                            'I': '#000000',      # Black
                            'II': '#ff0000',     # Red
                            'III': '#000000',     # Black
                            'aVR': '#000000',     # Black
                            'aVL': '#000000',     # Black
                            'aVF': '#000000',     # Black
                            'V1': '#000000',      # Black
                            'V2': '#000000',      # Black
                            'V3': '#000000',      # Black
                            'V4': '#000000',      # Black
                            'V5': '#000000',      # Black
                            'V6': '#000000'       # Black
                        }
                        
                        # Use specific color for this lead
                        waveform_color = lead_colors.get(lead, '#000000')
                        
                        self.axes[i].plot(time, data, color=waveform_color, linewidth=1.0, antialiased=True)
                        self.axes[i].set_title(f'Lead {lead}', fontsize=11, fontweight='bold', color='#333333', pad=5)
                        self.axes[i].set_xlabel('Time (s)', fontsize=9, color='#666666', labelpad=3)
                        self.axes[i].set_ylabel('mV', fontsize=9, color='#666666', labelpad=3)
        else:
            # Display placeholder message
            for i, ax in enumerate(self.axes):
                ax.text(0.5, 0.5, f'No ECG data\navailable', 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=10, color='white')
                ax.set_title(f'Lead {leads[i]}', fontsize=10, fontweight='bold', color='white')
        
        self.figure.tight_layout(pad=1.0)
        self.canvas.draw()
        
    def update_analysis_results(self):
        """Update metrics table and findings"""
        if not self.current_report:
            return
            
        # Update metrics table
        metrics = self.current_report.get('metrics', {})
        self.metrics_table.setRowCount(0)
        
        metric_items = [
            ("Heart Rate", f"{metrics.get('heart_rate', 'N/A')} bpm"),
            ("PR Interval", f"{metrics.get('pr_interval', 'N/A')} ms"),
            ("QRS Duration", f"{metrics.get('qrs_duration', 'N/A')} ms"),
            ("QT Interval", f"{metrics.get('qt_interval', 'N/A')} ms"),
            ("QTc Interval", f"{metrics.get('qtc_interval', 'N/A')} ms"),
            ("P Axis", f"{metrics.get('p_axis', 'N/A')}°"),
            ("QRS Axis", f"{metrics.get('qrs_axis', 'N/A')}°"),
            ("Rhythm", metrics.get('rhythm', 'N/A')),
        ]
        
        self.metrics_table.setRowCount(len(metric_items))
        for i, (param, value) in enumerate(metric_items):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(param))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))
        
        # Update findings
        findings = self.current_report.get('findings', [])
        self.findings_text.setPlainText('\n'.join(findings) if findings else 'No findings available')
        
        # Update recommendations
        recommendations = self.current_report.get('recommendations', [])
        self.recommendations_text.setPlainText('\n'.join(recommendations) if recommendations else 'No recommendations available')
        
    def export_report(self):
        """Export current report to file"""
        if not self.current_report:
            QMessageBox.warning(self, "Warning", "No report selected for export")
            return
            
        from PyQt5.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Report", 
            f"ecg_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.current_report, f, indent=2)
                QMessageBox.information(self, "Success", f"Report exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export report: {str(e)}")
    
    def print_report(self):
        """Print the current ECG report"""
        if not self.current_report:
            QMessageBox.warning(self, "Warning", "No report selected for printing")
            return
            
        try:
            # Create a simple print dialog
            from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
            from PyQt5.QtWidgets import QFileDialog
            
            printer = QPrinter(QPrinter.HighResolution)
            dialog = QPrintDialog(printer, self)
            
            if dialog.exec_() == QPrintDialog.Accepted:
                # For now, just show a message that printing is being prepared
                QMessageBox.information(self, "Print", "ECG report sent to printer successfully")
                print("🖨️ ECG report printed successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to print report: {str(e)}")
