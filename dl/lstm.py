# 股票每日开盘价、最高、最低、收盘、成交量的数据集
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
from mpl_finance import candlestick_ohlc
from pylab import mpl
from datetime import datetime
import tensorflow as tf
from sdps_ds import SdpsDs

class Lstm(object):
    rnn_unit = 10         # 隐层神经元的个数
    lstm_layers = 2       # 隐层层数
    input_size = 7        # 输入层神经元个数
    output_size = 1       # 输出层神经元个数
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
    def create_lstm(X):
        print('初始化LSTM网络')
        batch_size = tf.shape(X)[0]
        time_step = tf.shape(X)[1]
        w_in = Lstm.weights['in']
        b_in = Lstm.biases['in']
        input = tf.reshape(X, [-1, Lstm.input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
        input_rnn = tf.matmul(input, w_in) + b_in
        input_rnn = tf.reshape(input_rnn, [-1, time_step, Lstm.rnn_unit])  #将tensor转成3维，作为lstm cell的输入
        cell = tf.nn.rnn_cell.MultiRNNCell([Lstm.create_lstm_cell() for i in range(Lstm.lstm_layers)])
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        output_rnn,final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
        output = tf.reshape(output_rnn, [-1, Lstm.rnn_unit]) 
        w_out = Lstm.weights['out']
        b_out = Lstm.biases['out']
        pred = tf.matmul(output, w_out) + b_out
        return pred, final_states
        
    @staticmethod
    def create_lstm_cell():
        #basicLstm单元
        basicLstm = tf.nn.rnn_cell.BasicLSTMCell(Lstm.rnn_unit)
        # dropout
        drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=Lstm.keep_prob)
        return basicLstm
      
    @staticmethod
    def train(batch_size=60, time_step=20, train_begin=2000, train_end=5800):
        X = tf.placeholder(tf.float32, shape=[None, time_step, Lstm.input_size])
        Y = tf.placeholder(tf.float32, shape=[None, time_step, Lstm.output_size])
        batch_index,train_x,train_y = SdpsDs.get_train_ds(batch_size, time_step, train_begin, train_end)
        with tf.variable_scope("sec_lstm"):
            pred,_=Lstm.create_lstm(X)
        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1])-tf.reshape(Y, [-1])))
        train_op = tf.train.AdamOptimizer(Lstm.lr).minimize(loss)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(10):     
                for step in range(len(batch_index)-1):
                    _,loss_= sess.run([train_op, loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]], Lstm.keep_prob:0.5})
                print("Number of iterations:",i," loss:",loss_)
            print("保存模型参数: ",saver.save(sess,'./model_save2/modle.ckpt'))
            print("模型训练结束")
            
            
            
            
            
            
    def predict(time_step=20):
        X = tf.placeholder(tf.float32, shape=[None, time_step, Lstm.input_size])
        mean,std,test_x,test_y = SdpsDs.get_test_ds(time_step)
        with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
            pred, _ = Lstm.create_lstm(X)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            #参数恢复
            module_file = tf.train.latest_checkpoint('model_save2')
            saver.restore(sess, module_file)
            test_predict=[]
            for step in range(len(test_x)-1):
              prob = sess.run(pred, feed_dict={X:[test_x[step]], Lstm.keep_prob:1})
              predict = prob.reshape((-1))
              test_predict.extend(predict)
            test_y = np.array(test_y)*std[7]+mean[7]
            test_predict = np.array(test_predict)*std[7]+mean[7]
            acc = np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差程度
            print("测试样本集精度:",acc)
            #以折线图表示结果
            plt.figure()
            plt.plot(list(range(len(test_predict))), test_predict, color='b',)
            plt.plot(list(range(len(test_y))), test_y,  color='r')
            plt.show()
        
    
