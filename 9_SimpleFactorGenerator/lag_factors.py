import pandas as pd

list = [1, 3, 5, 7, 9, 10, 20, 30, 60]

data = pd.DataFrame()


def lag_calculator(returns, laglist):
    for n in laglist:
        col = 'lag_{}'.format(n)
        data[col] = returns.shift(n)
    return data
