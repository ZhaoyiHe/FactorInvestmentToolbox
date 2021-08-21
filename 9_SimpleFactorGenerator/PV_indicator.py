import pandas as pd


def PV_indicator_calculator(stock_info, mom_range=None, window_size_range=None):
    """
    Calculate several price/volume time series indicators
    :param stock_info: pd.DataFrame()
    :param mom_range: list
    :param window_size_range: list
    :return: result:pd.DataFrame()
    """

    result = pd.DataFrame()
    close, volume, high, low = stock_info['close'], stock_info['volume'], stock_info['high'], stock_info['low']
    for i in mom_range:
        col_p = 'mom_price_{}'.format(i)
        result[col_p] = close / close.shift(i) - 1
        col_v = 'mom_volume_{}'.format(i)
        result[col_v] = volume / volume.shift(i) - 1
    for j in window_size_range:
        col_pl = 'rolling_price_low'.format(j)
        result[col_pl] = low.rolling(window=j).min()
        col_ph = 'rolling_price_high'.format(j)
        result[col_ph] = high.rolling(window=j).max()
        col_vl = 'rolling_volume_low'.format(j)
        result[col_vl] = volume.rolling(window=j).min()
        col_vh = 'rolling_volume_high'.format(j)
        result[col_vh] = volume.rolling(window=j).max()

        col_pr = 'rolling_price_ratio'.format(j)
        result[col_pr] = (close - close.shift(1)) / (
                    high.rolling(j).max() - low.rolling(j).min())  # (p_t - p_{t-1})/(ph_lambda - pl_lambda)
        col_vr = 'rolling_volume_ratio'.format(j)
        result[col_vr] = (volume - volume.shift(1)) / (volume.rolling(j).max() - volume.rolling(j).min())

    return result


if __name__ == "__main__":
    data = pd.read_csv('sample_data.csv')
    mom_list = [1, 3, 5, 7, 9]
    window_list = [3, 5, 10, 20]
    result = PV_indicator_calculator(data, mom_list, window_list)
