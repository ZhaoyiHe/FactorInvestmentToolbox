import pandas as pd


def frequency_trans(data):
    monthly_data = data.resample('M', on='date').last()
    return monthly_data


if __name__ == "__main__":
    data = pd.read_csv('sample_data.csv')
    data = data.reset_index()
    data = data.rename(columns={"order_book_id": "symbol"})
    data['date'] = pd.to_datetime(data.date)
    monthly_data = frequency_trans(data)
    print(monthly_data)
