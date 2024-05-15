import sys
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Oblicza odchylenie standardowe portfela
def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)


# Oblicza oczekiwaną stopę zwrotu
def expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights) * 252


# Oblicza wskaźnik Sharpe'a
def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)


# Definiuje funkcję do minimalizacji (negatywny wskaźnik Sharpe'a)
def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)


# Pobiera ceny zamknięcia dla podanych tickerów z Yahoo Finance
def download_close_prices(tickers, start_date, end_date):
    adj_close_df = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        adj_close_df[ticker] = data["Adj Close"]
    return adj_close_df


# Pobiera aktualną rentowność 10-letnich obligacji skarbowych USA
def get_risk_free_rate():
    treasury_data = yf.download("^TNX", period="1d")
    risk_free_rate = treasury_data["Close"].iloc[-1] / 100
    return risk_free_rate


# Tworzy wykres optymalnych wag portfela
def plot_optimal_weights(tickers, optimal_weights):
    plt.figure(figsize=(10, 6))
    plt.bar(tickers, optimal_weights)
    plt.xlabel("Aktywa")
    plt.ylabel("Optymalne wagi")
    plt.title("Optymalne wagi portfela")
    plt.show()


# Funkcja do obliczenia i wyświetlenia maksymalnych skumulowanych zwrotów
def calculate_maximum_return(tickers, start_date, end_date):
    adj_close_df = download_close_prices(tickers, start_date, end_date)
    daily_returns = adj_close_df.pct_change().dropna()
    cumulative_returns = (1 + daily_returns).cumprod()
    max_returns = cumulative_returns.max()
    return max_returns.items()


def optimize_portfolio(tickers, bounds, start_date, end_date):
    adj_close_df = download_close_prices(tickers, start_date, end_date)
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
    cov_matrix = log_returns.cov() * 252
    risk_free_rate = get_risk_free_rate()

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    initial_weights = np.array([1 / len(tickers)] * len(tickers))

    optimized_results = minimize(
        neg_sharpe_ratio,
        initial_weights,
        args=(log_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
    )

    return optimized_results.x


# Funkcja tworząca wykres
def create_figure(tickers, optimal_weights):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(tickers, optimal_weights)
    ax.set_xlabel("Aktywa")
    ax.set_ylabel("Optymalne wagi")
    ax.set_title("Optymalne wagi portfela")
    return fig
