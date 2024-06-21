import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns, BlackLittermanModel, HRPOpt, black_litterman


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


def create_figure(tickers, optimal_weights):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(tickers, optimal_weights)
    ax.set_xlabel("Aktywa")
    ax.set_ylabel("Optymalne wagi")
    ax.set_title("Optymalne wagi portfela")
    return fig


# Helper function - market cap
def fetch_market_caps(tickers):
    market_caps = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        market_cap = stock.info.get('marketCap', 0)
        market_caps.append(market_cap)
    return market_caps


def optimize_black_litterman(tickers, views, start_date, end_date):

    adj_close_df = download_close_prices(tickers, start_date, end_date)

    S = risk_models.sample_cov(adj_close_df)

    market_caps = fetch_market_caps(tickers)
    market_caps_series = pd.Series(market_caps, index=tickers)

    delta = black_litterman.market_implied_risk_aversion(adj_close_df.mean())
    prior = black_litterman.market_implied_prior_returns(market_caps_series, delta, S)

    Q, P = views["Q"], views["P"]

    bl = BlackLittermanModel(S, pi=prior, Q=Q, P=P)

    rets = bl.bl_returns()

    ef = EfficientFrontier(rets, S)
    ef.add_constraint(lambda w: w >= 0.05)  # Minimum 5% alokacji
    ef.add_constraint(lambda w: w <= 0.4)   # Maximum 40% alokacji
    ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    return cleaned_weights


def optimize_risk_parity(tickers, start_date, end_date):
    try:
        adj_close_df = download_close_prices(tickers, start_date, end_date)
        adj_close_df = adj_close_df.dropna()

        # NaN lub nieskonczonosc
        if not np.isfinite(adj_close_df.values).all():
            raise ValueError("Price data contains NaN or infinite values.")

        log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

        # NaN lub nieskonczonosc
        if not np.isfinite(log_returns.values).all():
            raise ValueError("Log returns contain NaN or infinite values.")

        log_returns = log_returns.astype(float)
        cov_matrix = risk_models.sample_cov(log_returns)

        # NaN lub nieskonczonosc
        if not np.isfinite(cov_matrix.values).all():
            #print("Covariance matrix contains NaN or infinite values before correction.")
            cov_matrix = cov_matrix.replace([np.inf, -np.inf], np.nan).dropna(how='any')
            #print("Covariance matrix after correction:\n", cov_matrix)

        if cov_matrix.empty or cov_matrix.isnull().values.any():
            # w najgorszm przypadku macierz jednostkowa zamiast covariancji
            cov_matrix = pd.DataFrame(np.identity(len(tickers)), index=tickers, columns=tickers)
            print("Fallback identity matrix...")

        # Normalize the covariance matrix
        cov_matrix /= np.diag(cov_matrix).mean()

        # NaN lub nieskonczonosc na wszelki wypadek
        if not np.isfinite(cov_matrix.values).all():
            raise ValueError("Covariance matrix still contains NaN or infinite values after normalization...")

        hrp = HRPOpt(cov_matrix)
        optimal_weights = hrp.optimize()

        #print("Optymalne:", optimal_weights)

        # ewentualne ograncizenia
        for key in optimal_weights:
            if optimal_weights[key] > 0.8:
                optimal_weights[key] = 0.8

        # normalizacja wag
        total_weight = sum(optimal_weights.values())
        optimal_weights = {k: v / total_weight for k, v in optimal_weights.items()}

        return optimal_weights
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}
