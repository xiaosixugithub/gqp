import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf
from sdps_ds import SdpsDs
from lstm import Lstm

class SppEngine(object):
    def __init__(self):
        self.name = 'SppEngine'

    def startup(self):
        print('股票价格预测引擎......')
        Lstm.train()
        Lstm.predict()
