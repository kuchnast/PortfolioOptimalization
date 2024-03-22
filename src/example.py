import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize
from fredapi import Fred
import matplotlib.pyplot as plt

FRED_API_KEY = "d1e52014cca3903e340c5ed8480b79f6"


# Calculate the portfolio standard deviation
def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)


# Calculate the expected return
# Key Assumption: Expected returns are based on historical returns
def expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights) * 252


# Calculate the Sharpe Ratio
def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)


# Define the function to minimize (negative Sharpe Ratio)
def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)


def download_close_prices(tickers, start_date, end_date):
    # Create an empty DataFrame to store the adjusted close prices
    adj_close_df = pd.DataFrame()

    # Download the close prices for each ticker
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        adj_close_df[ticker] = data["Adj Close"]

    return adj_close_df


def plot_optimal_weights(tickers, optimal_weights):
    # Create a bar chart of the optimal weights
    plt.figure(figsize=(10, 6))
    plt.bar(tickers, optimal_weights)

    # Add labels and a title
    plt.xlabel("Assets")
    plt.ylabel("Optimal Weights")
    plt.title("Optimal Portfolio Weights")

    # Display the chart
    plt.show()


def main():
    tickers = ["SPY", "BND", "GLD", "QQQ", "VT"]

    # Set the end date to today
    end_date = datetime.today()

    # Set the start date to 5 years ago
    start_date = end_date - timedelta(days=5 * 365)
    print(start_date)

    adj_close_df = download_close_prices(tickers, start_date, end_date)

    # Display the dataframe
    print(adj_close_df)

    # Calculate the lognormal returns for each ticker
    log_returns = np.log(adj_close_df / adj_close_df.shift(1))

    # Drop any missing values
    log_returns = log_returns.dropna()

    # Calculate the covariance matrix using annualized log returns
    cov_matrix = log_returns.cov() * 252
    print(cov_matrix)

    fred = Fred(api_key=FRED_API_KEY)
    ten_year_treasury_rate = fred.get_series_latest_release("GS10") / 100

    # Set the risk-free rate
    risk_free_rate = ten_year_treasury_rate.iloc[-1]
    print(risk_free_rate)

    # Set the constraints and bounds
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    bounds = [(0, 0.5) for _ in range(len(tickers))]

    # Set the initial weights
    initial_weights = np.array([1 / len(tickers)] * len(tickers))
    print(initial_weights)

    # This is assuming 'neg_sharpe_ratio', 'initial_weights', 'log_returns', 'cov_matrix', 'risk_free_rate',
    # 'constraints', and 'bounds' have been defined previously.
    optimized_results = minimize(
        neg_sharpe_ratio,
        initial_weights,
        args=(log_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
    )

    # Get the optimal weights
    optimal_weights = optimized_results.x

    # Display analytics of the optimal portfolio
    print("Optimal Weights:")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight:.4f}")

    print()

    optimal_portfolio_return = expected_return(optimal_weights, log_returns)
    optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
    optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

    print(f"Expected Annual Return: {optimal_portfolio_return:.6f}")
    print(f"Expected Volatility: {optimal_portfolio_volatility:.4f}")
    print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")

    plot_optimal_weights(tickers, optimal_weights)


if __name__ == "__main__":
    main()
