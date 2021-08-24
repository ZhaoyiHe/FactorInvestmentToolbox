import os
import numpy as np
import pandas as pd
import rqdatac as rq
from rqdatac import *


def get_industry_info(stock_returns, industry_df):
    industry_info_df = pd.DataFrame(index=stock_returns.unstack().index).reset_index().rename(columns = {"level_0":"symbol"})
    # dates = stock_returns.reset_index().date.tolist()
    # initialize a dataframe
    # industry_info_df = pd.DataFrame(index=stock_returns.index)
    # industry_info_df = industry_info_df.reset_index()
    # industry_info_df.columns = ['symbol', 'date']
    # extract trading dates
    stock_returns = stock_returns.reset_index()
    dates = stock_returns.date.unique()
    # extract industry code
    codes = industry_df.code.to_list()

    for i in range(len(codes)):

        print(" Processing on industry %d" % codes[i])
        industry_info = dict()
        for j in range(len(dates)):
            print("     Processing on date %s" % dates[j])
            result = get_industry(str(codes[i]), source='citics_2019', date=dates[j])

            industry_info[dates[j]] = result
        industry_info_agg = pd.DataFrame.from_dict(industry_info, orient="index")
        industry_info_agg = industry_info_agg.stack().reset_index().drop('level_1', axis=1)
        industry_info_agg.columns = ['date', 'symbol']
        industry_info_agg[codes[i]] = 1

        industry_info_df = pd.merge(industry_info_df, industry_info_agg, how='outer').fillna(0)

    industry_info_df = industry_info_df.sort_values(['date', 'symbol']).set_index(['date', 'symbol'])
    return industry_info_df


# industry_index_list = industry_df.symbol.to_list()
# industry_index_price = rq.get_price(industry_index_list,start_date=date[0],end_date=date[-1])

if __name__ == "__main__":
    rq.init()
    os.chdir('/Users/zoey/PycharmProjects/FactorInvestment/FactorInvestmentToolbox/2_EffectiveFactorRecognition')
    industry_df = pd.read_csv('./data/citics_industry.csv')
    stock_returns = pd.read_pickle('./data/stock_returns.pkl')
    # print(get_industry_info(stock_returns, industry_df))
    industry_info_df = get_industry_info(stock_returns, industry_df)
    industry_info_df.to_pickle('./data/industry_info_df.pkl')
