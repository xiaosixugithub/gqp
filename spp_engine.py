import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf
from sdps_ds import SdpsDs

class SppEngine(object):
    def __init__(self):
        self.name = 'SppEngine'

    def startup(self):
        print('股票价格预测引擎......')
        #SdpsDs.draw_kline()
        batch_index,train_x,train_y = SdpsDs.get_train_data()
        print(batch_index)
        print(train_x)
        print(train_y)
