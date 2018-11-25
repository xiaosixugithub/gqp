# 股票每日开盘价、最高、最低、收盘、成交量的数据集
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
from mpl_finance import candlestick_ohlc
from pylab import mpl
from datetime import datetime

class SdpsDs(object):
    ds_file = 'dataset_2.csv'
    
    def __init__(self):
        self.cls_name = 'SdpsDs'
       
    @staticmethod
    def get_train_ds(batch_size=60,time_step=20,train_begin=0,train_end=5800):
        '''
        获取训练样本集
        '''
        data = SdpsDs.initialize_data()
        batch_index = []
        data_train = data[train_begin:train_end]
        normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
        train_x,train_y=[],[]   #训练集
        for i in range(len(normalized_train_data)-time_step):
           if i % batch_size==0:
               batch_index.append(i)
           x=normalized_train_data[i:i+time_step,:7]
           y=normalized_train_data[i:i+time_step,7,np.newaxis]
           train_x.append(x.tolist())
           train_y.append(y.tolist())
        batch_index.append((len(normalized_train_data)-time_step))
        return batch_index,train_x,train_y
        
    @staticmethod
    def get_test_ds(time_step=20,test_begin=5800):
        '''
        获取训练样本集
        '''
        data = SdpsDs.initialize_data()
        data_test=data[test_begin:]
        mean=np.mean(data_test,axis=0)
        std=np.std(data_test,axis=0)
        normalized_test_data=(data_test-mean)/std  #标准化
        size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
        test_x,test_y=[],[]
        for i in range(size-1):
           x=normalized_test_data[i*time_step:(i+1)*time_step,:7]
           y=normalized_test_data[i*time_step:(i+1)*time_step,7]
           test_x.append(x.tolist())
           test_y.extend(y)
        test_x.append((normalized_test_data[(i+1)*time_step:,:7]).tolist())
        test_y.extend((normalized_test_data[(i+1)*time_step:,7]).tolist())
        return mean,std,test_x,test_y

        
    @staticmethod
    def initialize_data():
        '''
        从原始数据文件中读出股票数据
        '''
        with open(SdpsDs.ds_file) as fd:
            df = pd.read_csv(fd)     #读入股票数据
            data = df.iloc[:,2:10].values  #取第3-10列
        return data

    @staticmethod
    def draw_kline():
        print('绘制股票每日价格K线图')
        raw_datas = pd.read_csv('kline.csv')
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
        ax.set_title(u'上证综指k线图')
        candlestick_ohlc(ax, recs, width=0.7, colorup='r', colordown='g')
        plt.setp(plt.gca().get_xticklabels(), rotation=50, horizontalalignment='center')
        plt.show()
