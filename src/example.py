import sys
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QMessageBox,
    QListWidget,
    QDialog,
    QDialogButtonBox,
    QTextEdit,
)

from calculate import *


class CryptoSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_tickers = []
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        popular_cryptos = [
            ("Bitcoin", "BTC-USD"),
            ("Ethereum", "ETH-USD"),
            ("Binance Coin", "BNB-USD"),
            ("Tether", "USDT-USD"),
            ("Solana", "SOL-USD"),
            ("Cardano", "ADA-USD"),
            ("XRP", "XRP-USD"),
            ("Polkadot", "DOT-USD"),
            ("Dogecoin", "DOGE-USD"),
            ("Avalanche", "AVAX-USD"),
            ("Shiba Inu", "SHIB-USD"),
            ("Terra", "LUNA-USD"),
            ("Chainlink", "LINK-USD"),
            ("Litecoin", "LTC-USD"),
            ("Algorand", "ALGO-USD"),
            ("Stellar", "XLM-USD"),
            ("VeChain", "VET-USD"),
            ("Tron", "TRX-USD"),
            ("EOS", "EOS-USD"),
            ("Monero", "XMR-USD"),
        ]
        for name, ticker in popular_cryptos:
            self.list_widget.addItem(f"{name} ({ticker})")
        layout.addWidget(self.list_widget)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def accept(self):
        selected_items = self.list_widget.selectedItems()
        self.selected_tickers = [item.text().split(" (")[1][:-1] for item in selected_items]
        super().accept()

    def reject(self):
        self.selected_tickers = []
        super().reject()


class CryptoPortfolioOptimizer(QWidget):
    def __init__(self):
        super().__init__()
        self.ticker_entries = []
        self.lower_entries = []
        self.upper_entries = []
        self.results_text = QTextEdit(self)
        self.results_text.setReadOnly(True)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        tickers_layout = QHBoxLayout()
        tickers_label = QLabel("Tickery:")
        tickers_layout.addWidget(tickers_label)
        self.ticker_entries = [QLineEdit(self) for _ in range(5)]
        for ticker_entry in self.ticker_entries:
            button = QPushButton("Wybierz", self)
            button.clicked.connect(lambda _, e=ticker_entry: self.show_crypto_selection_dialog(e))
            tickers_layout.addWidget(ticker_entry)
            tickers_layout.addWidget(button)
        layout.addLayout(tickers_layout)

        bounds_layout = QHBoxLayout()
        lower_label = QLabel("Dolny limit:")
        bounds_layout.addWidget(lower_label)
        self.lower_entries = [QLineEdit(self) for _ in range(5)]
        for lower_entry in self.lower_entries:
            bounds_layout.addWidget(lower_entry)

        upper_label = QLabel("Górny limit:")
        bounds_layout.addWidget(upper_label)
        self.upper_entries = [QLineEdit(self) for _ in range(5)]
        for upper_entry in self.upper_entries:
            bounds_layout.addWidget(upper_entry)
        layout.addLayout(bounds_layout)

        optimize_button = QPushButton("Optymalizuj portfel", self)
        optimize_button.clicked.connect(self.optimize_portfolio)
        layout.addWidget(optimize_button)

        calculate_max_button = QPushButton("Oblicz maksymalne zwroty", self)
        calculate_max_button.clicked.connect(self.calculate_maximum_return)
        layout.addWidget(calculate_max_button)

        layout.addWidget(self.results_text)

        self.setLayout(layout)
        self.setWindowTitle("Optymalizacja portfela kryptowalut")

    def show_crypto_selection_dialog(self, entry):
        dialog = CryptoSelectionDialog(self)
        if dialog.exec_():
            selected_tickers = dialog.selected_tickers
            entry.setText(", ".join(selected_tickers))

    def optimize_portfolio(self):
        tickers = []
        for ticker_entry in self.ticker_entries:
            if ticker_entry.text():
                tickers.extend(ticker_entry.text().split(", "))
        bounds = [
            (float(lower_entry.text()), float(upper_entry.text()))
            for lower_entry, upper_entry in zip(self.lower_entries, self.upper_entries)
            if lower_entry.text() and upper_entry.text()
        ]
        end_date = datetime.today() - timedelta(days=365)
        start_date = end_date - timedelta(days=5 * 365)
        optimal_weights = optimize_portfolio(tickers, bounds, start_date, end_date)
        optimal_weights_msg = "\n".join([f"{ticker}: {weight:.4f}" for ticker, weight in zip(tickers, optimal_weights)])
        self.results_text.setText(f"Optymalne wagi:\n{optimal_weights_msg}")

        fig = create_figure(tickers, optimal_weights)
        self.display_plot(fig)

    def calculate_maximum_return(self):
        tickers = []
        for ticker_entry in self.ticker_entries:
            if ticker_entry.text():
                tickers.extend(ticker_entry.text().split(", "))
        start_date = datetime.today() - timedelta(days=365)
        end_date = datetime.today()
        max_returns = calculate_maximum_return(tickers, start_date, end_date)
        max_return_msg = "\n".join([f"{ticker}: {max_return:.4f}" for ticker, max_return in max_returns])
        self.results_text.setText(f"Maksymalne skumulowane zwroty:\n{max_return_msg}")

    def display_plot(self, fig):
        canvas = FigureCanvas(fig)
        canvas.draw()
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(canvas)
        self.setLayout(plot_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = CryptoPortfolioOptimizer()
    ex.show()
    sys.exit(app.exec_())
