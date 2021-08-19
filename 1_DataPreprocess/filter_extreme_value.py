import numpy as np


def filter_extreme_values(series, n=3):
    '''
    用中位数去极值
    '''
    median = series.quantile(0.5)
    md = ((series - median).abs()).quantile(0.5)
    max_limit = median + n * md
    min_limit = median - n * md
    return (np.clip(series, min_limit, max_limit))
