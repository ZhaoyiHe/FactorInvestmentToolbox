def standardize(series):
    mean = series.mean()
    std = series.std()
    return ((series - mean) / std)


def rank_standardize(series):
    series = series.rank()
    mean = series.mean()
    std = series.std()
    return ((series - mean) / std)
