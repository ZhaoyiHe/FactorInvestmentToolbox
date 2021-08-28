def standardize(series):
    """Standardize series/Calculate Z-score"""
    mean = series.mean()
    std = series.std()
    return ((series - mean) / std)


def rank_standardize(series):
    """Standardize rank of series"""
    series = series.rank()
    mean = series.mean()
    std = series.std()
    return ((series - mean) / std)
