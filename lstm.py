# 股票每日开盘价、最高、最低、收盘、成交量的数据集
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
from mpl_finance import candlestick_ohlc
from pylab import mpl
from datetime import datetime
import tensorflow as tf

class Lstm(object):
    rnn_unit = 10         #隐层神经元的个数
    lstm_layers = 2       #隐层层数
    input_size = 7
    output_size = 1
    lr = 0.0006         #学习率
    weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
    biases={
            'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
            'out':tf.Variable(tf.constant(0.1,shape=[1,]))
           }
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')    
    
    @staticmethod
    def initialize():
        print('初始化LSTM网络')