def standardize(series):
    '''
    进行标准化处理
    '''

    mean = series.mean()
    std = series.std()
    return ((series - mean) / std)