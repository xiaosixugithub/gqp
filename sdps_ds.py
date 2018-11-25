# 股票每日开盘价、最高、最低、收盘、成交量的数据集
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SdpsDs(object):
    def __init__(self):
        self.cls_name = 'SdpsDs'

    @staticmethod
    def draw_kline():
        print('绘制股票每日价格K线图')
        datas = pd.read_csv('dataset_2.csv')
        datas = datas.iloc[:, 1:]
        print(datas[-30:, :])
        
