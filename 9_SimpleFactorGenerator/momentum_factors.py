import numpy as np
import pandas as pd

list = [1, 3, 5, 7, 9, 10, 20, 30, 60]

data = pd.DataFrame()


def mom_calculator(returns, momlist):
    for n in momlist:
        col = 'mom_{}'.format(n)
        data[col] = np.sign(returns.rolling(n).mean())
    return data
