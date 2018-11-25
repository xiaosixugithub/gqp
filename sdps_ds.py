# 股票每日开盘价、最高、最低、收盘、成交量的数据集
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
from mpl_finance import candlestick_ohlc
from pylab import mpl
from datetime import datetime

class SdpsDs(object):
    def __init__(self):
        self.cls_name = 'SdpsDs'

    @staticmethod
    def draw_kline():
        print('绘制股票每日价格K线图')
        raw_datas = pd.read_csv('dataset_2.csv')
        datas = raw_datas.iloc[-30:, 1:]
        datas.date = [date2num( datetime.strptime(date, '%Y/%m/%d') ) for date in datas.date]
        recs = list()
        for i in range(len(datas)):
            recs.append(datas.iloc[i, :])
        ax = plt.subplot()
        mondays = WeekdayLocator(MONDAY)
        weekFormatter = DateFormatter('%y %b %d')
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(DayLocator())
        ax.xaxis.set_major_formatter(weekFormatter)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        ax.set_title(u'上证综指kline picture')
        candlestick_ohlc(ax, recs, width=0.7, colorup='r', colordown='g')
        plt.setp(plt.gca().get_xticklabels(), rotation=50, horizontalalignment='center')
        plt.show()
