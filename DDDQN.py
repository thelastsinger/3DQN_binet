#!/usr/bin/python3.6
# -*- encoding: utf-8 -*-

# 网络结构定义，接受sumo里的数据矩阵为输入，输出动作q值
# The class of dueling DQN with three convolutional layers

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim

np.random.seed(1)
tf.set_random_seed(1)
# 神经网络
class Qnetwork():
    def __init__(self,h_size,action_num,n):
        # 从sumo获取一个状态，展平成一个数组,60行 * 60列
        # 修改输入的结构后，通过三个卷积层进行处理
        # self.scalarInput = tf.placeholder(shape=[None, 14400, 2], dtype=tf.float32)  # 大地图原状态输入
        self.scalarInput = tf.placeholder(shape=[None, 120, 4], dtype=tf.float32)      #大地图低精度带转向


        # self.scalarInput =  tf.placeholder(shape=[None,600,2],dtype=tf.float32) #输入
        # self.scalarInput = tf.placeholder(shape=[None, 3600, 2], dtype=tf.float32) # 论文里的输入
        # self.scalarInput = tf.placeholder(shape=[None, 60, 2], dtype=tf.float32)    # 低精度输入
        # self.scalarInput = tf.placeholder(shape=[None, 60, 4], dtype=tf.float32) # 低精度加转向输入
        self.legal_actions =  tf.placeholder(shape=[None,action_num],dtype=tf.float32) #行为空间
        # self.imageIn = tf.reshape(self.scalarInput,shape=[-1,20,30,2])  # 把输入的结构修改（原本的结构大小）
        # self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 60, 60, 2]) # 论文里的结构大小
        # self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 4, 15, 2])    # 非精确的结构大小

        # self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 120, 120, 2])  # 大地图的结构大小
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 4, 30, 4])  # 大地图低精度的结构大小
        # slim.conv2d自带卷积功能+激活函数,其中slim可以使建立\训练和评估神经网络更加简单
        # 第一层卷积层c，卷积核个数（即输出）32，大小为8*8，步长为4

        #将输入转化成60*60的矩阵，作为卷积神经网络的输入，slim能更简洁的实现网络的定义，激活函数都是relu
        #一共3个卷积层，输出维度为128
        with tf.variable_scope('cov_1'+str(n)):
            self.conv1 = slim.conv2d( \
                inputs=self.imageIn, num_outputs=4, kernel_size=[2, 2], stride=[1, 1], padding='VALID',activation_fn=tf.nn.relu, biases_initializer=None)
                # inputs=self.imageIn,num_outputs=32,kernel_size=[4,4],stride=[2,2],padding='VALID', activation_fn=tf.nn.relu, biases_initializer=None)
        with tf.variable_scope('cov_2'+str(n)):
            self.conv2 = slim.conv2d( \
                inputs=self.conv1,num_outputs=8,kernel_size=[2,2],stride=[1,1],padding='VALID', activation_fn=tf.nn.relu, biases_initializer=None)
        with tf.variable_scope('cov3'+str(n)):
            self.conv3 = slim.conv2d( \
                inputs=self.conv2,num_outputs=16,kernel_size=[2,2],stride=[1,1],padding='VALID', activation_fn=tf.nn.relu, biases_initializer=None)
        '''

        '''
        # 将输出扁平化但保留batch_size,假设第一维是batch
        self.stream = slim.flatten(self.conv3) # 将conv3的输出值从三维变为1维的数组，保留batch_size，假设第一维是batch
        # 两个参数分别为网络输入\输出的神经元数量


        '''
        网络分为两个，一个控制ns绿灯时间一个控制we绿灯时间
        简化相位为两个，ns绿灯和we绿灯
        在ns绿灯结束的时候，在控制we绿灯时间的网络选取动作(+5s,-5s,和不变)，设置we绿灯时间，切换为we绿灯相位
        在we绿灯结束的时候同理
        训练的时候，分两个memory，一个存ns绿灯期间的(s,a,s'，r),另一个存we的。
        --！！--在step里得到的下一个状态s'是另一个相位的开始，所以是相互交换s',并且要一直保存一个相位的s',所以要一次计算三个step进行保存，或这次计算，下次保存
        '''



        with tf.variable_scope('fully_connected'+str(n)):  # Dueling前的全连接网络
            self.stream0 = slim.fully_connected(self.stream, 32, activation_fn=tf.nn.relu) # 输出单元为128个


        # with tf.variable_scope('LSTM_network'+str(n)):       # LSTM长短期记忆网络
        #     lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(128)
        #     self.stream0 = tf.expand_dims(self.stream0, axis=2)
        #     _, state1 = tf.nn.dynamic_rnn(lstm_cell, self.stream0, dtype=tf.float32)
        #     self.stream0n = state1.h
        #
        #     print(self.stream.shape)
        #     print(state1.c)
        #     print(state1.h)


        # 之后分为action网络和value网络
        self.streamA = self.stream0
        self.streamV = self.stream0

        # 两个网络的输出输入尺寸相同
        with tf.variable_scope('Advantage'+str(n)):
            self.streamA0 = slim.fully_connected(self.streamA,h_size, activation_fn=tf.nn.relu)
        with tf.variable_scope('Value'+str(n)):
            self.streamV0 = slim.fully_connected(self.streamV, h_size, activation_fn=tf.nn.relu)
        
        xavier_init = tf.contrib.layers.xavier_initializer()            # 权重的初始化器,这个初始化器是用来保持每一层的梯度大小都差不多相同
        action_num = np.int32(action_num)

        self.AW = tf.Variable(xavier_init([h_size,action_num]))         # 设置Advantage的权重
        self.VW = tf.Variable(xavier_init([h_size,1]))                  # 设置分析state的value的权重,h_size表示几个状态,它所对应的值

        # 计算动作优势和状态价值
        self.Advantage = tf.matmul(self.streamA0,self.AW)
        self.Value = tf.matmul(self.streamV0,self.VW)
        
        # 采用优势函数的平均值代替上述的最优值,采用这种方法,虽然使得值函数V和优势函数A不再完美的表示值函数和优势函数(在语义上的表示),但是这种操作提高了稳定性
        # 在实际中,一般要将动作优势流设置为单独动作优势函数减去某状态下所有动作的优势函数的平均值,这样做可以保证该状态下各个动作的优势函数相对排序不变,而且
        # 可以缩小Q值的范围,去除多余的自由度,提高算法的稳定性

        # 计算总的Q值
        self.Qout0 = self.Value  + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keepdims=True))

        # 总的Q值还要减去非法action的惩罚值
        self.Qout = tf.add(self.Qout0, self.legal_actions)

        # 预测动作选择最大值
        # 预测的动作,argmax中的0:按列计算  1:按行计算
        self.predict = tf.argmax(self.Qout,1)   # tf.argmax(input, axis=None, name=None, dimension=None)，axis为1则为按行取最大的那个数的序号
        
        # 目标Q值和动作是更新时候的输入
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        # self.targetQ = tf.placeholder(tf.float32, [None, n_classes])
        # self.actions = tf.placeholder(tf.float32, [None, n_classes])

        # action的独热矩阵
        self.actions_onehot = tf.one_hot(self.actions,np.int32(action_num),dtype=tf.float32)
        # 对应的当前的Q值等于Qout乘以独热矩阵
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        # 设置td误差，loss和反向更新方法
        with tf.variable_scope('loss'):
            self.td_error = tf.square(self.targetQ - self.Q)
            self.loss = tf.reduce_mean(self.td_error)
        with tf.variable_scope('train'):
            self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.updateModel = self.trainer.minimize(self.loss)
        
    # def relu(self, x, alpha=0.01, max_value=None):
    #     '''ReLU.这个函数也可自己定义
    #
    #     alpha: slope of negative section.
    #     '''
    #     negative_part = tf.nn.relu(-x)
    #     x = tf.nn.relu(x)
    #     if max_value is not None:
    #         #tf.clip_by_value(A, min, max)输出一个张量,把A中的每一个元素的值都压缩到min和max之间
    #         x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
    #                              tf.cast(max_value, dtype=tf.float32))
    #     x -= tf.constant(alpha, dtype=tf.float32) * negative_part#若为正数,则输出该数,若为负数,输出一个很小的正数,缩小一点
    #     return x
    
    def choose_action(self, epsilon, observation, legal_action, sess, flag):
        #Choose an action (0-8) by greedily (with e chance of random action) from the Q-network
        #用epsilon选择动作的策略

        if np.random.uniform() < epsilon or flag == False:                  # 如果随机值小于e，就随机选择动作
            a_selected = [x for x in legal_action if x!=-1]     # 选择动作的范围是不是-1的所有动作
            a_num = len(a_selected)                             # 可选动作的数量
            a = np.random.randint(0, a_num)                     # 在0-动作数量之间随机选择一个数字
            action = a_selected[a]                              # 选择对应的动作
        else:
            # np.reshape(observation, [-1,600,2])    # 对应不同输入
            # np.reshape(observation, [-1, 60, 2])
            # np.reshape(observation, [-1, 14400, 2])
            # np.reshape(observation, [-1, 3600, 2])
            # np.reshape(observation, [-1, 60, 4])

            np.reshape(observation, [-1, 120, 4])

            legal_a_one = [0 if x!=-1 else -99999 for x in legal_action] #非-1的动作对应的值为0，-1的动作对应的值为一个很大的负数
            action = sess.run(self.predict,feed_dict={self.scalarInput:observation, self.legal_actions:[legal_a_one]})[0] #返回预测动作的值
        return action

