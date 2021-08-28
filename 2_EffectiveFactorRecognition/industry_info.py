import os

import pandas as pd
import rqdatac as rq
from rqdatac import *


def get_industry_info(stock_returns, industry_df):
    industry_info_df = pd.DataFrame(index=stock_returns.unstack().index).reset_index().rename(
        columns={"level_0": "symbol"})
    # extract trading dates
    stock_returns = stock_returns.reset_index()
    dates = stock_returns.date.unique()
    # extract industry codes
    codes = industry_df.code.to_list()
    # get industry components on every date
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


if __name__ == "__main__":
    rq.init()
    os.chdir('/Users/zoey/PycharmProjects/FactorInvestment/FactorInvestmentToolbox/2_EffectiveFactorRecognition')
    industry_df = pd.read_csv('./data/citics_industry.csv')
    stock_returns = pd.read_pickle('./data/stock_returns.pkl')
    industry_info_df = get_industry_info(stock_returns, industry_df)
    industry_info_df.to_pickle('./data/industry_info_df.pkl')
