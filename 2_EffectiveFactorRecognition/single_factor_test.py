import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

import data_preprocess


class SingleFactorTest(object):
    def __init__(self, factor_data, stock_returns, dates, stocks):
        """

        :param factor_data: single factor data
        :param stock_returns: all stock returns data in selected duration
        """
        self.factor_data = factor_data
        self.stock_returns = stock_returns
        self.dates = dates
        self.stocks = stocks

    def factor_dist_plot(self, filter_extreme_values=False):
        """
        Plot factor value distribution by year.
        :param filter_extreme_values: bool value
        """
        factor_data = self.factor_data.copy()
        factor_data = factor_data.stack().reset_index().rename(columns={0: "factor_value"})
        factor_data = factor_data.reset_index()
        factor_data['date'] = pd.to_datetime(factor_data.date)
        factor_data['year'] = factor_data.date.apply(lambda x: datetime.datetime.date(x).year)
        factor_data = factor_data.set_index(['date', 'symbol'])
        years = factor_data.year.unique()
        nrows = 3
        n = len(years)
        ncols = int(n / 3) + 1
        pre = data_preprocess.DataPreprocess()
        if filter_extreme_values:
            factor_data = factor_data.groupby('year').apply(
                lambda x: pre.filter_extreme_values(x['factor_value'])).reset_index()
        else:
            factor_data = factor_data.groupby('year').apply(lambda x: x['factor_value']).reset_index()
        factor_data = factor_data.drop(['date', 'symbol'], axis=1)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 8))
        for i in range(len(years)):
            sns.kdeplot(factor_data.loc[factor_data.year == years[i], 'factor_value'], shade=True,
                        ax=axes[int(i / ncols), i % ncols],
                        label=years[i])

    def single_factor_regression(self, industry_info, industry_index_returns):
        # dates = self.stock_returns.reset_index().date.to_list()
        # stocks = self.factor_data.columns
        factor_return = list()
        for i in range(0, len(self.dates) - 1):
            date = self.dates[i]
            print("Processing data on %s" % date)
            factor_date_temp = self.factor_data.loc[date, :]
            info = industry_info[industry_info['date'] == date]
            info = info.drop(['date'], axis=1).set_index(['symbol'])
            temp_ind = industry_index_returns.loc[date, :]
            temp = info.apply(lambda x: x.values * temp_ind, axis=1).fillna(0).apply(sum, axis=1)
            # np.where(test.all(axis=0) != 0)[0][0]
            # ind = pd.DataFrame(temp).iloc[:,np.where(temp.all(axis=0) != 0)[0][0]]
            stock_return_temp = self.stock_returns.iloc[i + 1, :]
            factor_date_temp = pd.DataFrame(factor_date_temp).reset_index()
            factor_date_temp.columns = ['symbol', 'factor_value']
            temp = pd.DataFrame(temp).reset_index()
            temp.columns = ['symbol', 'industry_return']
            stock_return_temp = pd.DataFrame(stock_return_temp).reset_index()
            stock_return_temp.columns = ['symbol', 'stock_return']
            data_agg = pd.merge(factor_date_temp, temp, how="outer")
            data_agg = pd.merge(data_agg, stock_return_temp, how="outer")
            data_agg = data_agg.dropna(axis=0)
            x = np.vstack((np.ones(len(data_agg)), data_agg.industry_return, data_agg.factor_value)).T
            y = data_agg.stock_return

            model = sm.OLS(y, x)

            f_returns = model.fit().params.x2
            factor_return.append(f_returns)

        factor_return = pd.DataFrame(index=self.dates[:-1], data=factor_return)
        return factor_return

    def single_factor_WLS_regression(self, mktcap_data):

        factor_return = list()
        for i in range(0, len(self.dates) - 1):
            date = self.dates[i]
            print("Processing data on %s" % date)
            factor_date_temp = self.factor_data.loc[date, :]
            # info = industry_info[industry_info['date'] == date]
            # info = info.drop(['date'], axis=1).set_index(['symbol'])
            # temp_ind = industry_index_returns.loc[date, :]
            # temp = info.apply(lambda x: x.values * temp_ind, axis=1).fillna(0).apply(sum, axis=1)
            # # np.where(test.all(axis=0) != 0)[0][0]
            # ind = pd.DataFrame(temp).iloc[:,np.where(temp.all(axis=0) != 0)[0][0]]
            stock_return_temp = self.stock_returns.iloc[i + 1, :]
            factor_date_temp = pd.DataFrame(factor_date_temp).reset_index()
            factor_date_temp.columns = ['symbol', 'factor_value']

            # temp = pd.DataFrame(temp).reset_index()
            # temp.columns = ['symbol', 'industry_return']
            stock_return_temp = pd.DataFrame(stock_return_temp).reset_index()
            stock_return_temp.columns = ['symbol', 'stock_return']
            mktcap_data_temp = mktcap_data.loc[date, :]
            weights = mktcap_data_temp.apply(np.sqrt).fillna(0).reset_index()
            weights.columns = ['symbol', 'weights']
            # data_agg = pd.merge(factor_date_temp, temp, how="outer")
            data_agg = pd.merge(factor_date_temp, weights, how="outer")
            data_agg = pd.merge(data_agg, stock_return_temp, how="outer")
            data_agg = data_agg.dropna(axis=0)
            x = np.vstack((np.ones(len(data_agg)), data_agg.factor_value)).T
            y = data_agg.stock_return

            model = sm.WLS(y, x, weights=data_agg.weights)
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

    def industry_norm(self):
        # stocks = self.factor_data.columns
        factor_data = dict()
        for i in range(0, len(self.dates) - 1):
            date = self.dates[i]
            print("Norm:Processing data on %s" % date)
            factor_date_temp = self.factor_data.loc[date, :]
            info = industry_info[industry_info['date'] == date]
            info = info.drop(['date'], axis=1)
            info = info.set_index(['symbol'])
            temp_ind = industry_index_returns.iloc[(i + 1), :]
            temp = info.apply(lambda x: x.values * temp_ind, axis=1).fillna(0).apply(sum, axis=1)
            # np.where(test.all(axis=0) != 0)[0][0]
            # ind = pd.DataFrame(temp).iloc[:,np.where(temp.all(axis=0) != 0)[0][0]]
            # stock_return_temp = self.stock_returns.loc[date, :]
            factor_date_temp = pd.DataFrame(factor_date_temp).reset_index()
            factor_date_temp.columns = ['symbol', 'factor_value']
            temp = pd.DataFrame(temp).reset_index()
            temp.columns = ['symbol', 'industry_return']
            # stock_return_temp = pd.DataFrame(stock_return_temp).reset_index()
            # stock_return_temp.columns = ['symbol', 'stock_return']
            data_agg = pd.merge(factor_date_temp, temp, how="outer")
            # data_agg = pd.merge(data_agg, stock_return_temp, how="outer")
            data_agg = data_agg.dropna(axis=0)
            x = np.vstack((np.ones(len(data_agg)), data_agg.industry_return)).T
            y = data_agg.factor_value

            model = sm.OLS(y, x)
            factor_data[date] = model.fit().resid
        normalized_factor_data = pd.DataFrame.from_dict(factor_data).T
        normalized_factor_data.columns = stocks
        self.normalized_factor_data = normalized_factor_data

    def IC_calculator(self):

        # stocks = self.factor_data.columns
        IC_value = list()

        for i in range(0, len(self.dates) - 1):
            date = self.dates[i]
            print("IC:Processing data on %s" % date)
            factor_date_temp = self.normalized_factor_data.loc[date, :]

            stock_return_temp = self.stock_returns.iloc[i + 1, :]
            factor_date_temp = pd.DataFrame(factor_date_temp).reset_index()
            factor_date_temp.columns = ['symbol', 'factor_value']
            stock_return_temp = pd.DataFrame(stock_return_temp).reset_index()
            stock_return_temp.columns = ['symbol', 'stock_return']

            data_agg = pd.merge(factor_date_temp, stock_return_temp, how="outer")
            data_agg = data_agg.dropna(axis=0)
            x = np.vstack((np.ones(len(data_agg)), data_agg.factor_value)).T
            y = data_agg.stock_return

            IC = stats.spearmanr(y, x)[0]
            IC_value.append(IC)

        IC_value = pd.DataFrame(index=dates[:-1], data=IC_value)
        return IC_value

    def ICIR_calculator(self):
        IC_series = self.IC_calculator()
        IC_mean = np.mean(IC_series)
        IC_std = np.std(IC_series)
        ICIR = IC_mean / IC_std
        return ICIR

    def factor_backtesting_by_groups(self, groups_number=10,head=None):
        group_return_tot = dict()
        for i in range(0, len(self.dates) - 1):
            date = self.dates[i]
            print("Backtesting:Processing data on %s" % date)
            factor_date_temp = self.factor_data.loc[date, :]
            # info = industry_info[industry_info['date'] == date]
            # info = info.drop(['date'], axis=1)
            # info = info.set_index(['symbol'])
            # temp_ind = industry_index_returns.iloc[(i + 1), :]
            # temp = info.apply(lambda x: x.values * temp_ind, axis=1).fillna(0).apply(sum, axis=1)
            # np.where(test.all(axis=0) != 0)[0][0]
            # ind = pd.DataFrame(temp).iloc[:,np.where(temp.all(axis=0) != 0)[0][0]]
            stock_return_temp = self.stock_returns.iloc[(i + 1), :]
            factor_date_temp = pd.DataFrame(factor_date_temp).reset_index()
            factor_date_temp.columns = ['symbol', 'factor_value']
            # temp = pd.DataFrame(temp).reset_index()
            # temp.columns = ['symbol', 'industry_return']
            stock_return_temp = pd.DataFrame(stock_return_temp).reset_index()
            stock_return_temp.columns = ['symbol', 'stock_return']
            # data_agg = pd.merge(factor_date_temp, temp, how="outer")
            data_agg = pd.merge(factor_date_temp, stock_return_temp, how="outer")
            data_agg = data_agg.dropna(axis=0)
            data_agg['groups'] = np.ceil(data_agg['factor_value'].rank() / (len(data_agg) / groups_number))

            group_return = data_agg.groupby('groups').apply(lambda x: x['stock_return'].mean())
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
            group_return_cum.plot()
        return group_return_cum.iloc[-1, :]


if __name__ == "__main__":
    os.chdir('/Users/zoey/PycharmProjects/FactorInvestment/FactorInvestmentToolbox/2_EffectiveFactorRecognition')
    # input factor data
    factor_data = pd.read_pickle('./data/factor_data.pkl')
    # factor_data = factor_data.reset_index().rename(columns={"order_book_id":"symbol"})
    # stocks = factor_data.symbol.unique().tolist()
    # factor_name = factor_data.columns[-1]
    # factor_data = factor_data.sort_values(['date', 'symbol']).set_index(['date', 'symbol']).unstack()
    # factor_data.columns=stocks
    # apply data preprocess
    pre = data_preprocess.DataPreprocess()
    processed = factor_data.apply(lambda x: pre.standardize(pre.filter_extreme_values(x)), axis=1)  #####

    # input industrial data
    industry_info = pd.read_pickle('./data/industry_info_df.pkl')
    industry_index_returns = pd.read_pickle('./data/industry_index_returns.pkl')
    industry_info = industry_info.reset_index()
    # stocks = processed.symbol.unique().tolist()

    # input stock returns
    stock_returns = pd.read_pickle('./data/stock_returns.pkl')
    # stock_returns.columns = stock_returns.reset_index().stack().reset_index().symbol.unique().tolist()[1:]
    # dates = factor_data.reset_index().date.to_list()
    # stock_returns = stock_returns.reset_index()
    # stock_returns = stock_returns[stock_returns['date'].isin(dates)].set_index('date')
    stocks = factor_data.columns.to_list()
    dates = factor_data.reset_index().date.to_list()

    mktcap_data = pd.read_pickle('./data/mktcap_data.pkl')
    single_test = SingleFactorTest(factor_data, stock_returns, stocks=stocks, dates=dates)
    factor_return = single_test.single_factor_regression(industry_info=industry_info,
                                                         industry_index_returns=industry_index_returns)
    single_test.industry_norm()
    IC_series = single_test.IC_calculator()
    print(single_test.ICIR_calculator())
    single_test.factor_backtesting_by_groups(head=30)