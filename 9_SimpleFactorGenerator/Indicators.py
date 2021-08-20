import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rqdatac as rq


class IndicatorClass(object):
    def __init__(self):
        pass

    def SMA_generator(self, prices, period):
        sma = prices.close.rolling(period).mean()
        return sma

    def EMA_generator(self, prices, period):
        prices = prices.close
        ema = list()
        for i in range(len(prices)):
            if i == 0:
                ema.append(prices.iloc[0])
            else:
                smooth_coef = 2 / (period + 1)
                current_ema = smooth_coef * prices.iloc[i] + (1 - smooth_coef) * ema[i - 1]
                ema.append(current_ema)
        ema = pd.DataFrame(index=prices.index, data=ema)
        return ema

    def EMMA_generator(self, prices, period):
        emma = list()
        for i in range(len(prices)):
            if i == 0:
                emma.append(prices.iloc[0])
            else:
                smooth_coef = 1 / period
                current_ema = smooth_coef * prices.iloc[i] + (1 - smooth_coef) * emma[i - 1]
                emma.append(current_ema)
        emma = pd.DataFrame(index=prices.index, data=emma)
        return emma

    def RSI_cal(self, prices, period):

        U_list = prices[-1] - prices[:-1]

        D_list = prices[:-1] - prices[-1]
        U_list = pd.DataFrame(np.where(U_list > 0, U_list, 0))
        D_list = pd.DataFrame(np.where(D_list > 0, D_list, 0))

        U_emma = self.EMMA_generator(prices=U_list, period=period)
        D_emma = self.EMMA_generator(prices=D_list, period=period)
        if D_emma.values[-1] == 0:
            RSI_single = 100
        else:
            RS = U_emma.values[-1] / D_emma.values[-1]
            RSI_single = 100 - 100 / (1 + RS)
        return RSI_single

    def RSI_generator(self, prices, period):
        """

        :param prices:
        :param period: 6, 9, 14
        :return:
        """
        rsi = pd.Series(index=prices.index, dtype=float)
        for i in range(len(prices)):
            if i in range(period):
                rsi.iloc[i] = 100
            else:
                print(i)
                rsi.iloc[i] = self.RSI_cal(prices[:(i + 1)], period)
                print(rsi.iloc[i])
        return rsi

    def BOLL_generator(self, prices, period=None, k=None, term="medium"):
        if term == "short":
            period = 10
            k = 1.5
        elif term == "medium":
            period = 20
            k = 2
        elif term == "long":
            period = 50
            k = 2.5
        TP = (prices['high'] + prices['low'] + prices['close']) / 3
        MD = TP.rolling(window=period).mean()
        std = TP.rolling(window=period).std()
        UB = MD + k * std
        LB = MD - k * std
        result = pd.concat([UB, MD, LB], axis=1)
        result.columns = ['UB', 'MD', 'LB']
        return result

    def SAR_generator(self, prices):
        pass

    def BBI_generator(self, prices, period_set=None):
        if period_set is None:
            period_set = [3, 6, 12, 24]
        MAset = list()
        for i in range(len(period_set)):
            MAset.append(self.SMA_generator(prices.close, period_set[i]))
        bbi = pd.concat(MAset, axis=1).apply(np.mean, axis=1)
        return bbi

    def ROC_generator(self, prices, period):
        data = prices.close
        roc = (data - data.shift(period)) / data.shift(period)
        return roc

    def MACD_generator(self, prices, short=12, long=26):
        ema_short = self.EMA_generator(prices=prices, period=short)
        ema_long = self.EMA_generator(prices=prices, period=long)
        diff = ema_short - ema_long
        dea = list()
        for i in range(len(prices)):
            if i == 0:
                diff_init = diff.values[0][0]
                dea.append(diff_init)
            else:
                dea_current = 0.8 * dea[i - 1] + 0.2 * diff.values[i][0]
                dea.append(dea_current)
        dea = pd.DataFrame(index=diff.index, data=dea)
        macd = (diff - dea) * 2
        return macd

    def VOL_generator(self, prices, period=10):
        """

        :param prices:
        :param period: 10 ,30
        :return:
        """
        vol = prices.volume.rolling(period).mean()
        return vol

    def CCI_generator(self, prices, period=14, alpha=0.015):
        TP = (prices['high'] + prices['low'] + prices['close']) / 3
        MD = TP.rolling(window=period).mean()
        std = TP.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean())
        cci = (TP - MD) / (alpha * std)
        return cci

    def WR_generator(self, prices, period):
        period_min_low = prices['low'].rolling(period).min()
        period_max_high = prices['high'].rolling(period).max()
        wr = (period_max_high - prices['close']) * 100 / (period_max_high - period_min_low)
        return wr

    def WVAD_generator(self, prices, period=6):
        vad = (prices['close'] - prices['open']) / (prices['high'] - prices['low']) * prices['volume']
        wvad = vad.rolling(window=period).sum()
        return wvad

    def DMI_generator(self, prices):
        pass

    def PSY_generator(self, prices, period=12):
        returns = prices.close.pct_change()

        psy = returns.rolling(period).apply(lambda x: np.sum(x > 0))
        return psy

    def KDJ_generator(self, prices, period=9):
        """
        - AX = 当日的收盘价 - 9天中的最低价
        - BX = 9天中的最高价 - 9天中的最低价
        - RSV = AX ÷ BX × 100
        - 当日K值 ＝ 2／3前一日K值 + 1/3当日的RSV
        - 当日D值 ＝ 1／3当日K值 + 2/3前一日D值
        - J值 = 3D - 2K
        - 注：第一次计算时，前一日的K、D一律以50代替。
        :param prices:
        :return:
        """
        period_min_low = prices['low'].rolling(period).min()
        period_max_high = prices['high'].rolling(period).max()
        AX = prices['close'] - period_min_low
        BX = period_max_high - period_min_low
        RSV = AX / BX * 100
        RSV = RSV.fillna(100)
        K = list()
        D = list()
        for i in range(len(RSV)):

            if i == 0:
                K.append(50)
                D.append(50)
            else:
                K_current = (2 / 3) * K[i - 1] + (1 / 3) * RSV.iloc[i]
                K.append(K_current)
                D_current = (2 / 3) * D[i - 1] + (1 / 3) * K_current
                D.append(D_current)
        K = pd.Series(index=prices.index, data=K)
        D = pd.Series(index=prices.index, data=D)
        J = 3 * D - 2 * K
        KDJ = pd.concat([K, D, J], axis=1)
        KDJ.columns = ['K', 'D', 'J']
        return KDJ

    def OBV_generator(self, prices):
        diff = prices['close'] - prices['close'].shift(1)
        v_totay = prices['volume'] * np.sign(diff)
        v_totay.iloc[0] = prices['volume'].iloc[0]
        obv = v_totay.cumsum()
        return obv

    def VAO_generator(self, prices):
        """
        由于OBV的计算方法过于简单化，所以容易受到偶然因素的影响，为了提高OBV的准确性，可以采取多空比率净额法对其进行修正。
        多空比率净额= [（收盘价－最低价）－（最高价-收盘价）] ÷（ 最高价－最低价）×V
        该方法根据多空力量比率加权修正成交量，比单纯的OBV法具有更高的可信度。
        :param prices:
        :return:
        """

        v_totay = ((prices['close'] - prices['low']) - (prices['high'] - prices['close'])) * prices['volume'] / (
                prices['high'] - prices['low'])
        v_totay = v_totay.fillna(0)
        v_totay.iloc[0] = prices['volume'].iloc[0]
        vao = v_totay.cumsum()
        return vao


if __name__ == "__main__":
    rq.init()
    Ind = IndicatorClass()
    data = rq.get_price('300467.XSHE', '2015-05-01', '2021-08-06', frequency='1d', adjust_type="pre")
    wvad = Ind.WVAD_generator(data)
    wvad.plot()
    plt.show()
