import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import logging
import data_preprocess


class SingleFactorTest(object):
    def __init__(self, factor_data, stock_returns):
        """
        Process operations on the single factor.
        :param factor_data: single factor data
        :param stock_returns: all stock returns data in selected duration
        """
        self.factor_data = factor_data
        self.stock_returns = stock_returns
        self.dates = factor_data.reset_index().date.to_list()
        self.stocks = factor_data.columns.to_list()

    def factor_dist_plot(self, filter_extreme_values=None):
        """
        Plot factor value distribution by year.
        :param filter_extreme_values: bool value
        """
        factor_data_loc = self.factor_data.copy()
        factor_data_loc = factor_data_loc.stack().reset_index().rename(columns={0: "factor_value"})
        # factor_data_loc = factor_data_loc.reset_index()
        factor_data_loc['date'] = pd.to_datetime(factor_data_loc.date)
        factor_data_loc['year'] = factor_data_loc.date.apply(lambda x: datetime.datetime.date(x).year)
        factor_data_loc = factor_data_loc.set_index(['date', 'symbol'])
        years = factor_data_loc.year.unique()
        nrows = 3
        n = len(years)
        ncols = int(n / 3) + 1
        pre = data_preprocess.DataPreprocess()
        if filter_extreme_values:
            factor_data_loc = factor_data_loc.groupby('year').apply(
                lambda x: pre.filter_extreme_values(x['factor_value'])).reset_index()
        factor_data_loc = factor_data_loc.drop(['date', 'symbol'], axis=1)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 8))
        for i in range(len(years)):
            sns.kdeplot(factor_data_loc.loc[factor_data_loc.year == years[i], 'factor_value'], shade=True,
                        ax=axes[int(i / ncols), i % ncols],
                        label=years[i])
        plt.show()

    def single_factor_regression(self, industry_info, industry_index_returns):
        logging.basicConfig(filename='single_factor_test.log', format='%(levelname)s:%(message)s', level=logging.DEBUG)
        factor_return = list()
        for i in range(0, len(self.dates) - 1):

            date = self.dates[i]
            # print("Processing data on %s" % date)
            logging.info("Processing data on %s" % date)
            # get factor value on the given date
            factor_date_temp = self.factor_data.loc[date, :]
            factor_date_temp = pd.DataFrame(factor_date_temp).reset_index()
            factor_date_temp.columns = ['symbol', 'factor_value']
            # get industry returns on the given date
            industry_info_loc = industry_info[industry_info['date'] == date]
            industry_info_loc = industry_info_loc.drop(['date'], axis=1).set_index(['symbol'])
            temp_ind = industry_index_returns.iloc[i + 1, :]
            temp = industry_info_loc.apply(lambda x: x.values * temp_ind, axis=1).fillna(0).apply(sum, axis=1)
            temp = pd.DataFrame(temp).reset_index()
            temp.columns = ['symbol', 'industry_return']
            # get stock returns on the given date
            stock_return_temp = self.stock_returns.iloc[i + 1, :]
            stock_return_temp = pd.DataFrame(stock_return_temp).reset_index()
            stock_return_temp.columns = ['symbol', 'stock_return']
            # aggregate stock return, industry return and factor value data
            data_agg = pd.merge(factor_date_temp, temp, how="outer")
            data_agg = pd.merge(data_agg, stock_return_temp, how="outer")
            data_agg = data_agg.dropna(axis=0)
            # do linear regression and get factor value
            x = np.vstack((np.ones(len(data_agg)), data_agg.industry_return, data_agg.factor_value)).T
            y = data_agg.stock_return
            model = sm.OLS(y, x)
            f_returns = model.fit().params.x2
            factor_return.append(f_returns)

        factor_return = pd.DataFrame(index=self.dates[:-1], data=factor_return)
        return factor_return

    def single_factor_WLS_regression(self, mktcap_data):
        """ Use square root of market capitalization as weights, perform WLS"""
        logging.basicConfig(filename='single_factor_test.log', format='%(levelname)s:%(message)s', level=logging.DEBUG)
        factor_return = list()
        for i in range(0, len(self.dates) - 1):
            date = self.dates[i]
            # print("WLS:Processing data on %s" % date)
            logging.info("WLS:Processing data on %s" % date)
            # get factor value on the given date
            factor_date_temp = self.factor_data.loc[date, :]
            factor_date_temp = pd.DataFrame(factor_date_temp).reset_index()
            factor_date_temp.columns = ['symbol', 'factor_value']
            # get stock returns on the given date
            stock_return_temp = self.stock_returns.iloc[i + 1, :]
            stock_return_temp = pd.DataFrame(stock_return_temp).reset_index()
            stock_return_temp.columns = ['symbol', 'stock_return']
            # get market cap data
            mktcap_data_temp = mktcap_data.loc[date, :]
            weights = mktcap_data_temp.apply(np.sqrt).fillna(0).reset_index()
            weights.columns = ['symbol', 'weights']
            # aggregate stock return, factor value and market cap data
            data_agg = pd.merge(factor_date_temp, weights, how="outer")
            data_agg = pd.merge(data_agg, stock_return_temp, how="outer")
            data_agg = data_agg.dropna(axis=0)

            # apply WLS
            x = np.vstack((np.ones(len(data_agg)), data_agg.factor_value)).T
            y = data_agg.stock_return
            model = sm.WLS(y, x, weights=data_agg.weights)
            # algebra method
            # w = np.diag((1.0/np.sqrt(weights)))
            # w_inv = np.linalg.inv(w)
            # mat = np.linalg.inv(np.dot(np.transpose(x),w_inv),x))
            # param = np.dot(np.dot(np.dot(mat,np.transpose(x)),w_inv),y)

            f_returns = model.fit().params.x1
            factor_return.append(f_returns)

        factor_return = pd.DataFrame(index=self.dates[:-1], data=factor_return)
        return factor_return

    def factor_t_test(self, series):
        t_series = (series - series.mean()) / (series.std() / (np.sqrt(len(series) - 1)))
        mean_absolute_t = abs(t_series).mean()
        return mean_absolute_t

    def industry_norm(self,industry_info,industry_index_returns):
        """ Use industry return to normalize factor value"""
        logging.basicConfig(filename='single_factor_test.log', format='%(levelname)s:%(message)s', level=logging.DEBUG)
        # stocks = self.factor_data.columns
        factor_data = dict()

        for i in range(0, len(self.dates) - 1):
            print(i)
            date = self.dates[i]
            # print("Norm:Processing data on %s" % date)
            logging.info("Norm:Processing data on %s" % date)
            # get factor data
            factor_date_temp = self.factor_data.loc[date, :]
            factor_date_temp = pd.DataFrame(factor_date_temp).reset_index()
            factor_date_temp.columns = ['symbol', 'factor_value']
            # get industry returns
            industry_info_loc = industry_info[industry_info['date'] == date]
            industry_info_loc = industry_info_loc.drop(['date'], axis=1).set_index(['symbol'])
            temp_ind = industry_index_returns.iloc[(i + 1), :]
            temp = industry_info_loc.apply(lambda x: x.values * temp_ind, axis=1).fillna(0).apply(sum, axis=1)
            temp = pd.DataFrame(temp).reset_index()
            temp.columns = ['symbol', 'industry_return']
            # aggregate factor data and industry return data
            data_agg = pd.merge(factor_date_temp, temp, how="outer")
            data_agg = data_agg.dropna(axis=0)
            # get regression residual
            x = np.vstack((np.ones(len(data_agg)), data_agg.industry_return)).T
            y = data_agg.factor_value

            model = sm.OLS(y, x)
            factor_data[date] = model.fit().resid
        normalized_factor_data = pd.DataFrame.from_dict(factor_data).T
        normalized_factor_data.columns = self.stocks
        self.normalized_factor_data = normalized_factor_data

    def IC_calculator(self,industry_norm,**kargs):
        logging.basicConfig(filename='single_factor_test.log', format='%(levelname)s:%(message)s', level=logging.DEBUG)
        if industry_norm:
            self.industry_norm(industry_info,industry_index_returns)
            factor_data = self.normalized_factor_data
        else:
            factor_data = self.factor_data
        IC_value = list()
        rank_IC_value = list()
        for i in range(0, len(self.dates) - 1):
            date = self.dates[i]
            # print("IC:Processing data on %s" % date)
            logging.info("IC:Processing data on %s" % date)
            # get factor value
            factor_date_temp = factor_data.loc[date, :]
            factor_date_temp = pd.DataFrame(factor_date_temp).reset_index()
            factor_date_temp.columns = ['symbol', 'factor_value']
            # get stock return
            stock_return_temp = self.stock_returns.iloc[i + 1, :]
            stock_return_temp = pd.DataFrame(stock_return_temp).reset_index()
            stock_return_temp.columns = ['symbol', 'stock_return']
            # aggregate factor value and stock return
            data_agg = pd.merge(factor_date_temp, stock_return_temp, how="outer")
            data_agg = data_agg.dropna(axis=0)
            x = data_agg.factor_value
            y = data_agg.stock_return

            IC = stats.spearmanr(y, x)[0]
            IC_value.append(IC)

            rank_x = data_agg.factor_value.rank()
            rank_y = data_agg.stock_return.rank()

            rank_IC = stats.spearmanr(rank_y, rank_x)[0]
            rank_IC_value.append(rank_IC)
        IC_serieses = dict()
        IC_serieses['IC'] = pd.DataFrame(index=self.dates[:-1], data=IC_value)
        IC_serieses['rank_IC'] = pd.DataFrame(index=self.dates[:-1], data=rank_IC_value)
        self.IC_serieses = IC_serieses
        print('maxIC= %f' % abs(IC_serieses['IC']).max())
        return IC_serieses

    def ICIR_calculator(self,industry_norm=True,**kargs):
        if not hasattr(self,'IC_serieses'):
            self.IC_calculator(industry_norm,**kargs)

        def cal_ICIR(series):
            mean = np.mean(series)
            std = np.std(series)
            return mean/std

        ICIRs = dict()
        ICIRs['ICIR'] = cal_ICIR(self.IC_serieses['IC'])
        ICIRs['rank_ICIR'] = cal_ICIR(self.IC_serieses['rank_IC'])
        return ICIRs

    def factor_backtest_by_groups(self, groups_number=10, head=None):
        logging.basicConfig(filename='single_factor_test.log', format='%(levelname)s:%(message)s', level=logging.DEBUG)
        group_return_tot = dict()
        for i in range(0, len(self.dates) - 1):
            date = self.dates[i]
            logging.info('Backtesting:Processing data on %s'% date)
            # print("Backtesting:Processing data on %s" % date)
            # get factor value
            factor_date_temp = self.factor_data.loc[date, :]
            factor_date_temp = pd.DataFrame(factor_date_temp).reset_index()
            factor_date_temp.columns = ['symbol', 'factor_value']
            # get stock return
            stock_return_temp = self.stock_returns.iloc[(i + 1), :]
            stock_return_temp = pd.DataFrame(stock_return_temp).reset_index()
            stock_return_temp.columns = ['symbol', 'stock_return']
            # aggregate stock return and factor value
            data_agg = pd.merge(factor_date_temp, stock_return_temp, how="outer")
            data_agg = data_agg.dropna(axis=0)
            # divide data into groups
            data_agg['groups'] = np.ceil(data_agg['factor_value'].rank() / (len(data_agg) / groups_number))
            # equally buy stocks in each group
            group_return = data_agg.groupby('groups').apply(lambda x: x['stock_return'].mean())
            # Simply form a long-short portfolio
            if group_return.iloc[0] > group_return.iloc[-1]:
                group_return['LS'] = group_return.iloc[0] - group_return.iloc[-1]
            else:
                group_return['LS'] = group_return.iloc[-1] - group_return.iloc[0]
            group_return_tot[date] = group_return

        group_return_tot = pd.DataFrame.from_dict(group_return_tot).T
        group_return_cum = group_return_tot.apply(lambda x: (1 + x).cumprod() - 1, axis=0)
        if head is not None:
            group_return_cum.head(head).plot()
        else:
            plt.plot(group_return_cum)
            plt.show()
        return group_return_cum.iloc[-1, :]


if __name__ == "__main__":
    os.chdir('/Users/zoey/PycharmProjects/FactorInvestment/FactorInvestmentToolbox/2_EffectiveFactorRecognition')
    # input factor data
    factor_data = pd.read_pickle('./data/factor_data.pkl')



    pre = data_preprocess.DataPreprocess()
    processed = factor_data.apply(lambda x: pre.standardize(pre.filter_extreme_values(x)), axis=1)

    # input industrial data
    industry_info = pd.read_pickle('./data/industry_info_df.pkl')
    industry_index_returns = pd.read_pickle('./data/industry_index_returns.pkl')
    industry_info = industry_info.reset_index()

    # input stock returns
    stock_returns = pd.read_pickle('./data/stock_returns.pkl')

    mktcap_data = pd.read_pickle('./data/mktcap_data.pkl')
    # single_test = SingleFactorTest(factor_data, stock_returns)
    # single_test.factor_dist_plot(filter_extreme_values=True)

    single_test = SingleFactorTest(processed, stock_returns)
    # factor_return = single_test.single_factor_regression(industry_info=industry_info,industry_index_returns=industry_index_returns)
    # factor_return.plot()
    # (1+factor_return).cumprod().plot()
    # plt.show()
    # wls_factor_return = single_test.single_factor_WLS_regression(mktcap_data=mktcap_data)
    # wls_factor_return.plot()
    # (1+wls_factor_return).cumprod().plot()
    # plt.show()

    # IC_series = single_test.IC_calculator(industry_norm=True,industry_info=industry_info,industry_index_returns=industry_index_returns)
    #
    # ICIR = single_test.ICIR_calculator(industry_norm=True,industry_info=industry_info,industry_index_returns=industry_index_returns)

    # print(ICIR)
    backtest_result = single_test.factor_backtest_by_groups(head=100)
    print(backtest_result)