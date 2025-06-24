import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
# from matplotlib.finance import candlestick_ohlc  # Removed: not used and not available in recent matplotlib
import matplotlib.dates as mdates
from matplotlib.widgets import SpanSelector
import mplfinance as mpf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                             QTextEdit, QSplitter, QGroupBox, QGridLayout,
                             QScrollArea, QFrame, QMessageBox, QProgressBar,
                             QStatusBar, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Handles data loading and processing with optimization for large datasets"""
    
    @staticmethod
    def load_csv(file_path, sample_size=None):
        """Load CSV with optional sampling for large files"""
        try:
            # First, check file size and row count
            df = pd.read_csv(file_path, nrows=1000)  # Sample to check format
            
            # Load full dataset
            df = pd.read_csv(file_path)
            
            # Convert datetime column
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Validate required columns
            required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            # Sample data if too large for performance
            if sample_size and len(df) > sample_size:
                df = df.iloc[::len(df)//sample_size].reset_index(drop=True)
            
            return df
        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")
    
    @staticmethod
    def calculate_daily_volatility(df):
        """Calculate daily volatility using high-low range"""
        df = df.copy()
        df['daily_volatility'] = (df['high'] - df['low']) / df['close'] * 100
        return df
    
    @staticmethod
    def calculate_annualized_volatility(daily_vols):
        """Calculate annualized volatility from daily volatilities"""
        if len(daily_vols) == 0:
            return 0
        
        # Mean daily volatility
        mean_daily_vol = np.mean(daily_vols)
        
        # Annualize (assuming 252 trading days per year)
        annualized_vol = mean_daily_vol * np.sqrt(252)
        return annualized_vol

class InteractiveChart(FigureCanvas):
    """Custom chart widget with interactive selection capabilities"""
    
    selection_changed = pyqtSignal(object, object)  # start_date, end_date
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 8), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        self.df = None
        self.span_selector = None
        self.selected_range = None
        
        # Style the chart
        self.fig.patch.set_facecolor('#f0f0f0')
        self.ax.set_facecolor('#ffffff')
        
        # Enable interactive navigation
        self.setFocusPolicy(Qt.ClickFocus)
        self.setFocus()
        
    def plot_data(self, df):
        """Plot OHLC candlestick chart with interactive selection"""
        self.df = df.copy()
        self.ax.clear()
        
        if df is None or len(df) == 0:
            return
        
        # Prepare data for mplfinance-style plotting
        df_plot = df.set_index('datetime')
        
        # Create candlestick plot manually for better control
        self.plot_candlesticks(df)
        
        # Set up span selector for range selection
        self.setup_span_selector()
        
        # Format x-axis
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        self.fig.autofmt_xdate()
        
        # Labels and title
        self.ax.set_title('Stock Price - Drag to Select Date Range', fontsize=14, fontweight='bold')
        self.ax.set_ylabel('Price ($)', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
        # Enable zoom and pan
        self.ax.callbacks.connect('xlim_changed', self.on_xlims_change)
        
        self.draw()
    
    def plot_candlesticks(self, df):
        """Plot candlestick chart"""
        dates = mdates.date2num(df['datetime'])
        
        for i in range(len(df)):
            date = dates[i]
            open_price = df.iloc[i]['open']
            high_price = df.iloc[i]['high']
            low_price = df.iloc[i]['low']
            close_price = df.iloc[i]['close']
            
            # Color: green if close > open, red otherwise
            color = 'green' if close_price >= open_price else 'red'
            
            # Draw the high-low line
            self.ax.plot([date, date], [low_price, high_price], color='black', linewidth=1)
            
            # Draw the open-close box
            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)
            
            rect = Rectangle((date - 0.3, bottom), 0.6, height, 
                           facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            self.ax.add_patch(rect)
    
    def setup_span_selector(self):
        """Setup interactive span selector for date range selection"""
        def onselect(xmin, xmax):
            if self.df is None:
                return
            
            # Convert matplotlib dates back to datetime
            start_date = mdates.num2date(xmin)
            end_date = mdates.num2date(xmax)
            
            # Find closest dates in dataframe
            start_idx = np.argmin(np.abs(self.df['datetime'] - start_date))
            end_idx = np.argmin(np.abs(self.df['datetime'] - end_date))
            
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            
            actual_start = self.df.iloc[start_idx]['datetime']
            actual_end = self.df.iloc[end_idx]['datetime']
            
            self.selected_range = (start_idx, end_idx)
            self.selection_changed.emit(actual_start, actual_end)
        
        # Handle different matplotlib versions
        try:
            self.span_selector = SpanSelector(
                self.ax, onselect, 'horizontal',
                useblit=True, props=dict(alpha=0.3, facecolor='yellow'),
                span_stays=True
            )
        except TypeError:
            # Fallback for older matplotlib versions
            try:
                self.span_selector = SpanSelector(
                    self.ax, onselect, 'horizontal',
                    useblit=True, rectprops=dict(alpha=0.3, facecolor='yellow'),
                    span_stays=True
                )
            except TypeError:
                # Minimal version for compatibility
                self.span_selector = SpanSelector(
                    self.ax, onselect, 'horizontal',
                    useblit=True
                )
    
    def on_xlims_change(self, ax):
        """Handle zoom/pan events"""
        pass  # Can add additional functionality here if needed
    
    def get_selected_data(self):
        """Get the currently selected data range"""
        if self.selected_range is None or self.df is None:
            return None
        
        start_idx, end_idx = self.selected_range
        return self.df.iloc[start_idx:end_idx+1].copy()

class VolatilityCalculator(QThread):
    """Background thread for volatility calculations to prevent UI freezing"""
    
    calculation_complete = pyqtSignal(object, float)  # daily_vols, annualized_vol
    
    def __init__(self, selected_data):
        super().__init__()
        self.selected_data = selected_data
    
    def run(self):
        """Perform volatility calculations in background"""
        if self.selected_data is None or len(self.selected_data) == 0:
            self.calculation_complete.emit(pd.DataFrame(), 0.0)
            return
        
        # Calculate daily volatilities
        daily_vols = (self.selected_data['high'] - self.selected_data['low']) / self.selected_data['close'] * 100
        
        # Calculate annualized volatility
        annualized_vol = DataProcessor.calculate_annualized_volatility(daily_vols)
        
        # Create dataframe with results
        results_df = self.selected_data.copy()
        results_df['daily_volatility'] = daily_vols
        
        self.calculation_complete.emit(results_df, annualized_vol)

class StockVolatilityAnalyzer(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.df = None
        self.selected_data = None
        self.calculator_thread = None
        
        self.init_ui()
        self.setup_style()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Stock Volatility Analyzer')
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls and results
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Chart
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 1000])
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready - Load a CSV file to begin')
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def create_left_panel(self):
        """Create the left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File loading section
        file_group = QGroupBox("Data Import")
        file_layout = QVBoxLayout(file_group)
        
        self.load_button = QPushButton("Load CSV File")
        self.load_button.clicked.connect(self.load_csv_file)
        file_layout.addWidget(self.load_button)
        
        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setWordWrap(True)
        file_layout.addWidget(self.file_info_label)
        
        layout.addWidget(file_group)
        
        # Selection info section
        selection_group = QGroupBox("Selection Info")
        selection_layout = QVBoxLayout(selection_group)
        
        self.selection_info_label = QLabel("No range selected")
        self.selection_info_label.setWordWrap(True)
        selection_layout.addWidget(self.selection_info_label)
        
        layout.addWidget(selection_group)
        
        # Volatility results section
        results_group = QGroupBox("Volatility Analysis")
        results_layout = QVBoxLayout(results_group)
        
        # Annualized volatility display
        self.annualized_vol_label = QLabel("Annualized Volatility: -")
        self.annualized_vol_label.setFont(QFont("Arial", 12, QFont.Bold))
        results_layout.addWidget(self.annualized_vol_label)
        
        # Daily volatility table
        self.volatility_table = QTableWidget()
        self.volatility_table.setColumnCount(3)
        self.volatility_table.setHorizontalHeaderLabels(['Date', 'Close Price', 'Daily Vol (%)'])
        self.volatility_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.volatility_table.setMaximumHeight(300)
        results_layout.addWidget(self.volatility_table)
        
        layout.addWidget(results_group)
        
        # Statistics section
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("No data selected")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(stats_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self):
        """Create the right chart panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Chart
        self.chart = InteractiveChart()
        self.chart.selection_changed.connect(self.on_selection_changed)
        layout.addWidget(self.chart)
        
        # Chart controls
        controls_layout = QHBoxLayout()
        
        zoom_out_button = QPushButton("Zoom Out")
        zoom_out_button.clicked.connect(self.zoom_out)
        controls_layout.addWidget(zoom_out_button)
        
        reset_view_button = QPushButton("Reset View")
        reset_view_button.clicked.connect(self.reset_view)
        controls_layout.addWidget(reset_view_button)
        
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        return panel
    
    def setup_style(self):
        """Setup application styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QLabel {
                color: #333333;
            }
            QTableWidget {
                gridline-color: #cccccc;
                background-color: white;
            }
        """)
    
    def load_csv_file(self):
        """Load CSV file dialog and processing"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            self.status_bar.showMessage('Loading CSV file...')
            QApplication.processEvents()
            
            # Load data with sampling for large files
            self.df = DataProcessor.load_csv(file_path, sample_size=10000)
            
            # Update UI
            self.file_info_label.setText(
                f"Loaded: {len(self.df)} records\n"
                f"Date range: {self.df['datetime'].min().strftime('%Y-%m-%d')} to "
                f"{self.df['datetime'].max().strftime('%Y-%m-%d')}"
            )
            
            # Plot the data
            self.chart.plot_data(self.df)
            
            self.status_bar.showMessage('CSV loaded successfully')
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV file:\n{str(e)}")
            self.status_bar.showMessage('Error loading file')
        finally:
            self.progress_bar.setVisible(False)
    
    def on_selection_changed(self, start_date, end_date):
        """Handle chart selection changes"""
        if self.df is None:
            return
        
        # Update selection info
        self.selection_info_label.setText(
            f"Selected Range:\n{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )
        
        # Get selected data
        self.selected_data = self.chart.get_selected_data()
        
        if self.selected_data is None or len(self.selected_data) == 0:
            return
        
        # Start volatility calculation in background thread
        self.calculate_volatility()
    
    def calculate_volatility(self):
        """Calculate volatility in background thread"""
        if self.calculator_thread and self.calculator_thread.isRunning():
            self.calculator_thread.terminate()
            self.calculator_thread.wait()
        
        self.status_bar.showMessage('Calculating volatility...')
        self.calculator_thread = VolatilityCalculator(self.selected_data)
        self.calculator_thread.calculation_complete.connect(self.on_calculation_complete)
        self.calculator_thread.start()
    
    def on_calculation_complete(self, results_df, annualized_vol):
        """Handle completed volatility calculations"""
        if len(results_df) == 0:
            self.annualized_vol_label.setText("Annualized Volatility: No data")
            self.volatility_table.setRowCount(0)
            self.stats_label.setText("No data selected")
            return
        
        # Update annualized volatility display
        self.annualized_vol_label.setText(f"Annualized Volatility: {annualized_vol:.2f}%")
        
        # Update daily volatility table
        self.update_volatility_table(results_df)
        
        # Update statistics
        self.update_statistics(results_df)
        
        self.status_bar.showMessage(f'Analysis complete - {len(results_df)} days selected')
    
    def update_volatility_table(self, results_df):
        """Update the volatility results table"""
        self.volatility_table.setRowCount(len(results_df))
        
        for i, (_, row) in enumerate(results_df.iterrows()):
            date_item = QTableWidgetItem(row['datetime'].strftime('%Y-%m-%d'))
            price_item = QTableWidgetItem(f"${row['close']:.2f}")
            vol_item = QTableWidgetItem(f"{row['daily_volatility']:.2f}%")
            
            self.volatility_table.setItem(i, 0, date_item)
            self.volatility_table.setItem(i, 1, price_item)
            self.volatility_table.setItem(i, 2, vol_item)
    
    def update_statistics(self, results_df):
        """Update statistics display"""
        daily_vols = results_df['daily_volatility']
        
        stats_text = f"""
        Selected Period: {len(results_df)} days
        
        Daily Volatility Statistics:
        • Mean: {daily_vols.mean():.2f}%
        • Median: {daily_vols.median():.2f}%
        • Std Dev: {daily_vols.std():.2f}%
        • Min: {daily_vols.min():.2f}%
        • Max: {daily_vols.max():.2f}%
        
        Price Range:
        • Min Close: ${results_df['close'].min():.2f}
        • Max Close: ${results_df['close'].max():.2f}
        • Price Change: {((results_df['close'].iloc[-1] / results_df['close'].iloc[0] - 1) * 100):.2f}%
        """
        
        self.stats_label.setText(stats_text.strip())
    
    def zoom_out(self):
        """Zoom out on the chart"""
        if self.df is not None:
            self.chart.ax.set_xlim(self.df['datetime'].min(), self.df['datetime'].max())
            self.chart.draw()
    
    def reset_view(self):
        """Reset chart view to show all data"""
        if self.df is not None:
            self.chart.plot_data(self.df)

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Stock Volatility Analyzer")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = StockVolatilityAnalyzer()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()