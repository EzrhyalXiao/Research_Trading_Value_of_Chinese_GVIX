import numpy as np
from scipy.stats import norm
import pandas as pd
from tqdm import tqdm
from utils import get_backtest_info, bsm_option_price, prepare_data

# Main function to get backtest results
def get_backtest_result(sigma, thres, data, vix, commission):
    """
    Perform backtesting on the options data using the given sigma and thresholds.

    Parameters:
    sigma: list - a list of sigma column names in the VIX data.
    thres: list - a list of thresholds corresponding to each sigma.
    data: DataFrame - the original options data.
    vix: DataFrame - the VIX data.

    Returns:
    DataFrame - the results of the backtest.
    pnl_df: DataFrame - the profit and loss data.
    """
    result_df = pd.DataFrame()
    pnl_df = pd.DataFrame()
    shibor_file = 'data/shibor.csv'
    date_range = ('2015-02-09', '2024-04-30')

    for prepared_data, s in prepare_data(data, vix, sigma, shibor_file, date_range):
        # Calculate option price using Black-Scholes model
        prepared_data['price'] = 0
        for m in tqdm(range(len(prepared_data)), desc=f"Calculating prices for {s}"):
            prepared_data.loc[m, 'price'] = bsm_option_price(prepared_data.loc[m, 'opt'], prepared_data.loc[m, 'S0'],
                                                         prepared_data.loc[m, 'K'], prepared_data.loc[m, 'T'],
                                                         prepared_data.loc[m, 'sigma'], prepared_data.loc[m, 'rf'] / 100)

        # Generate signals based on price and thresholds
        prepared_data['signal'] = 0
        for j in tqdm(range(len(prepared_data)), desc=f"Generating signals for {s}"):
            if prepared_data.loc[j, 'price'] > (1 + thres[sigma.index(s)]) * prepared_data.loc[j, 'close']:
                prepared_data.loc[j, 'signal'] = 1
            elif prepared_data.loc[j, 'price'] < (1 - thres[sigma.index(s)]) * prepared_data.loc[j, 'close']:
                prepared_data.loc[j, 'signal'] = -1

        # Calculate returns
        prepared_data['return'] = prepared_data['ret'] * prepared_data['signal']
        dd = prepared_data.groupby('date')['return'].mean()
        pnl = (dd - commission).cumsum() + 1
        pnl = pnl.shift(1).fillna(1)
        pnl.index = pd.to_datetime(pnl.index)
        pnl_df[s] = pnl

        # Get backtest information
        res = pd.DataFrame(get_backtest_info(pnl), columns=[s])
        result_df = pd.concat([result_df, res], axis=1)

    return result_df, pnl_df