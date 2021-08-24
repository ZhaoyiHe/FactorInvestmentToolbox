import numpy as np


class DataPreprocess(object):
    def __init__(self):
        pass

    def filter_extreme_values(self,series, n=3):
        """
        Use median to filter extreme values
        """
        median = series.quantile(0.5)
        md = ((series - median).abs()).quantile(0.5)
        max_limit = median + n * md
        min_limit = median - n * md
        return np.clip(series, min_limit, max_limit)

    def standardize(self,series):
        mean = series.mean()
        std = series.std()
        return (series - mean) / std

    def rank_standardize(self,series):
        series = series.rank()
        mean = series.mean()
        std = series.std()
        return (series - mean) / std
