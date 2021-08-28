import random
import rqdatac as rq
import pandas as pd
import numpy as np
from rqdatac import *
import os


if __name__ == "__main__":
    os.chdir('/Users/zoey/PycharmProjects/FactorInvestment/FactorInvestmentToolbox/2_EffectiveFactorRecognition')
    rq.init()
    """
    Set parameters
    """
    index_symbol = '000906.XSHG'
    start_date = '2015-01-01'
    end_date = '2021-08-15'
    # all_factors = rq.get_all_factor_names()
    # selected_factors = list()
    # for i in range(100):
    #    selected_factors.append(random.choice(all_factors))
    selected_factor = 'peg_ratio_ttm'

    """
    Get historical index components.
    """
    print("Getting historical index components.")
    index_price = rq.get_price(index_symbol, start_date=start_date, end_date=end_date,
                               expect_df=True).close  # get index price
    dates = index_price.reset_index().date.unique()
    stocks_dict = dict()
    for date in dates:
        print(" Getting index components on %s ." % date)
        stocks_dict[date] = rq.index_components(index_symbol, date=date)  # get all index components on history
    stocks_df = pd.DataFrame.from_dict(stocks_dict)
    stocks_df = stocks_df.stack().reset_index().drop('level_0', axis=1)
    stocks_df.columns = ['date', 'symbol']
    all_stocks = stocks_df.symbol.unique().tolist()

    """
    Drop st stock and suspended data.
    """
    print("Drop st stock and suspended data.")
    is_st_data = rq.is_st_stock(all_stocks, start_date=start_date, end_date=end_date)
    is_not_st_data = is_st_data.stack().apply(lambda x: 1 if x is False else np.nan).dropna().reset_index().drop(0,
                                                                                                                 axis=1).rename(
        columns={"level_0": "date", "level_1": "symbol"})
    is_suspended_data = rq.is_suspended(all_stocks, start_date=start_date, end_date=end_date)
    is_not_suspended_data = is_suspended_data.stack().apply(
        lambda x: 1 if x is False else np.nan).dropna().reset_index().drop(0, axis=1).rename(
        columns={"level_0": "date", "level_1": "symbol"})
    available_data = pd.merge(is_not_st_data, is_not_suspended_data, how="inner")
    stocks_df = pd.merge(stocks_df, available_data, how="inner")
    all_stocks = stocks_df.symbol.unique().tolist()
    dates = stocks_df.date.unique().tolist()
    """
    Get stock returns.
    """
    print("Getting stock returns...")
    temp_price = rq.get_price(all_stocks, start_date=start_date, end_date=end_date, expect_df=True).close
    temp_price = temp_price.reset_index()
    temp_price.columns = ['symbol', 'date', 'price']
    data = pd.merge(stocks_df, temp_price, how="inner")
    data = data.sort_values(['date', 'symbol'])
    stock_returns = data.set_index(
        ['date', 'symbol']).unstack().pct_change()  # get all index components returns on history
    stock_returns.columns = all_stocks
    stock_returns.to_pickle('./data/stock_returns.pkl')

    """
    Get selected factor data.
    """
    print("Getting selected factor data.")
    factor_data = rq.get_factor(all_stocks, selected_factor, start_date=start_date,
                                end_date=end_date)  # get single_factor_data
    factor_data = factor_data.stack().reset_index().rename(
        columns={"level_0": "date", "level_1": "symbol", 0: "factor_value"})
    factor_data = pd.merge(factor_data, stocks_df, how="inner")
    factor_data = factor_data.pivot(index='date', columns='symbol', values='factor_value')
    factor_data.to_pickle('./data/factor_data.pkl')
    # selected_factors = rq.get_factor(index_comp,selected_factors,start_date='2015-01-01',end_date='2021-08-01')
    mktcap_data =rq.get_factor(all_stocks, 'market_cap', start_date=start_date,end_date=end_date)  # get single_factor_data
    mktcap_data = mktcap_data.stack().reset_index().rename(
        columns={"level_0": "date", "level_1": "symbol", 0: "mktcap_value"})
    mktcap_data = pd.merge(mktcap_data, stocks_df, how="inner")
    mktcap_data = mktcap_data.pivot(index='date', columns='symbol', values='mktcap_value')
    mktcap_data.to_pickle('./data/mktcap_data.pkl')
    """
    Get industry index returns.
    """
    print("Getting industry index returns...")
    industry_information = pd.read_csv('./data/citics_industry.csv')
    industry_index_list = industry_information.index_symbol.to_list()  # get list of industry index
    industry_index_price = get_price(industry_index_list, start_date=start_date, end_date=end_date,
                                     expect_df=True).close
    industry_index_price = industry_index_price.unstack().T
    industry_index_returns = industry_index_price.pct_change()  # get returns of industry indexes
    industry_index_returns.to_pickle('./data/industry_index_returns.pkl')
    #
    # """
    # Get dummy variables on index information of stocks.
    # """
    # print("Getting dummy variables on index information of stocks...")
    # stock_returns = pd.read_pickle('./data/stock_returns.pkl')
    # industry_information = pd.read_csv('./data/citics_industry.csv')
    # industry_info_df = pd.DataFrame(index=stock_returns.unstack().index).reset_index().drop('level_0',axis=1)
    # dates = stock_returns.reset_index().date.tolist()
    # # extract industry code
    # codes = industry_information.code.to_list()
    # for i in range(len(codes)):
    # print(" Processing on industry %d" % codes[i])
    # industry_info = dict()
    # for j in range(len(dates)):
    #     print("     Processing on date %s" % dates[j])
    #     result = get_industry(str(codes[i]), source='citics_2019', date=dates[j])
    #
    #     industry_info[dates[j]] = result
    # industry_info_agg = pd.DataFrame.from_dict(industry_info, orient="index")
    # industry_info_agg = industry_info_agg.stack().reset_index().drop('level_1', axis=1)
    # industry_info_agg.columns = ['date', 'symbol']
    # industry_info_agg[codes[i]] = 1
    #
    # industry_info_df = pd.merge(industry_info_df, industry_info_agg, how='outer').fillna(0)
    #
    # industry_info_df = industry_info_df.sort_values(['date', 'symbol']).set_index(['date', 'symbol'])
    # industry_info_df.to_pickle('./data/industry_info_df.pkl')
