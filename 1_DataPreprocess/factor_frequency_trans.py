import pandas as pd


def frequency_trans(data):
    """Transform the data frequency"""
    monthly_data = data.resample('M', on='date').last()
    return monthly_data

    # all_days = get_trading_dates('2020-03-31', '2021-06-30', market='cn')
    # dateRange = []
    # tempYear = None
    # dictYears = pd.DatetimeIndex(all_days).groupby(pd.DatetimeIndex(all_days).year)
    # for yr in dictYears.keys():
    #    tempYear = pd.DatetimeIndex(dictYears[yr]).groupby(pd.DatetimeIndex(dictYears[yr]).month)
    #    for m in tempYear.keys():
    #        dateRange.append(max(tempYear[m])) # Select the last available date

if __name__ == "__main__":
    data = pd.read_csv('sample_data.csv')
    data = data.reset_index()
    data = data.rename(columns={"order_book_id": "symbol"})
    data['date'] = pd.to_datetime(data.date)
    monthly_data = frequency_trans(data)
    print(monthly_data)
