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
        mean,std,test_x,test_y = SdpsDs.get_test_ds()
        print(test_x)
        print(test_y)
        print('mean={0}, {1}!'.format(mean, std))
