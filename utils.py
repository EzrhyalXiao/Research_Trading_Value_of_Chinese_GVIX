import numpy as np
from scipy.stats import norm
import pandas as pd
from tqdm import tqdm


def bsm_option_price(option_type, S0, K, T, sigma, rf):
    """
    Calculate the price of a European option using the Black-Scholes-Merton formula.

    Parameters:
        - option_type (str): Type of the option, 'call' or 'put'.
        - S0 (float): Current price of the underlying asset.
        - K (float): Strike price of the option.
        - T (float): Time to expiration of the option (in years).
        - sigma (float): Volatility of the underlying asset.
        - rf (float): Risk-free rate (annualized).

    Returns:
        - price (float): Price of the option.
    """
    d1 = (np.log(S0 / K) + (rf + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 0:
        price = S0 * norm.cdf(d1) - K * np.exp(-rf * T) * norm.cdf(d2)
    elif option_type == 1:
        price = K * np.exp(-rf * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    return price

def get_backtest_info(portfolio_value_ser):
    """
    Calculate various performance metrics for a given portfolio value series.

    Parameters:
    portfolio_value_ser (pd.Series): A pandas Series with portfolio values over time.

    Returns:
    pd.Series: A pandas Series containing performance metrics.
    """
    # Convert index to datetime and resample to daily values, dropping NaNs
    portfolio_value_ser.index = pd.to_datetime(portfolio_value_ser.index)
    portfolio_value_ser = portfolio_value_ser.resample('1D').last().dropna()

    # Calculate daily returns
    ret_ser = portfolio_value_ser.pct_change().dropna()

    # Calculate cumulative returns
    cum_ret_ser = (portfolio_value_ser.pct_change().fillna(0) + 0).cumprod()

    # Initialize a dictionary to store the results
    result = pd.Series(dtype='float64')

    # Calculate total return
    result['Total Return'] = cum_ret_ser.iloc[-1] - 1

    # Calculate annualized return
    result['Annualized Return'] = cum_ret_ser.iloc[-1] ** (252 / len(cum_ret_ser)) - 1

    # Calculate annualized volatility
    result['Annualized Volatility'] = ret_ser.std() * np.sqrt(252)

    # Calculate maximum drawdown
    result['Maximum Drawdown'] = max((np.maximum.accumulate(cum_ret_ser) - cum_ret_ser) / np.maximum.accumulate(cum_ret_ser))

    # Calculate Sharpe ratio
    result['Sharpe Ratio'] = result['Annualized Return'] / result['Annualized Volatility']

    # Calculate Calmar ratio
    result['Calmar Ratio'] = result['Annualized Return'] / result['Maximum Drawdown']

    # Find the end date of the maximum drawdown
    end = (cum_ret_ser.expanding().max() - cum_ret_ser).idxmax().strftime('%Y-%m-%d')
    result['Maximum Drawdown Start Date'] = (cum_ret_ser.loc[:end]).idxmax().strftime('%Y-%m-%d')
    result['Maximum Drawdown End Date'] = end

    # Format the results as percentages
    result['Total Return'] = '%.2f%%' % (result['Total Return'] * 100)
    result['Annualized Return'] = '%.2f%%' % (result['Annualized Return'] * 100)
    result['Annualized Volatility'] = '%.2f%%' % (result['Annualized Volatility'] * 100)
    result['Maximum Drawdown'] = '%.2f%%' % (result['Maximum Drawdown'] * 100)
    result['Sharpe Ratio'] = '%.2f' % (result['Sharpe Ratio'])
    result['Calmar Ratio'] = '%.2f' % (result['Calmar Ratio'])

    return result

def prepare_data(data, vix, sigma, shibor_file, date_range):
    """
    Prepare the data for backtesting by merging with VIX and Shibor data.

    Parameters:
    data: DataFrame - the original options data.
    vix: DataFrame - the VIX data.
    sigma: list - a list of sigma column names in the VIX data.
    shibor_file: str - the path to the Shibor data CSV file.
    date_range: tuple - the start and end dates for data filtering.

    Returns:
    DataFrame - the prepared data with additional columns.
    """
    shibor = pd.read_csv(shibor_file, encoding='gb18030').set_index('Unnamed: 0').loc[date_range]
    for i, s in enumerate(sigma):
        merged_data = pd.merge(data, vix[['asset_price', s]], left_index=True, right_index=True)
        merged_data = pd.merge(merged_data, shibor[['1']], left_index=True, right_index=True)
        merged_data.columns = ['code', 'close', 'exercise_date', 'K', 'opt_type', 'T-days', 'T', 'ret', 'S0', 'sigma', 'rf']
        merged_data['sigma'] = merged_data['sigma'] / 100
        merged_data['opt'] = merged_data['opt_type'].apply(lambda x: 0 if x == 'call' else 1)
        merged_data = merged_data[~merged_data['ret'].isna()]
        merged_data['date'] = merged_data.index
        yield merged_data.reset_index(drop=True), s