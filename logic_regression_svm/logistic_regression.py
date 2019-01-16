# -*- coding:utf-8 -*-
"""
Created on 18-12-9
Author: wbq813 (wbq813@foxmail.com)
Python3.6.6 deepin15.7
"""
import matplotlib.pyplot as plt
import math
import random
import numpy
from sklearn.datasets import load_svmlight_file


def process_data():
    x_train, y_train = load_svmlight_file("a9a", n_features=123)
    x_train = add_x_b(x_train.toarray())
    # 将具有-1和1的标签数组转化为0和1的标签数组
    y_train = numpy.array([int(y_item / 2 + 0.5) for y_item in y_train])
    # 将数据集的标签转化为列向量，方便使用矩阵乘法计算梯度
    y_train = y_train.reshape(len(y_train), 1)

    x_val, y_val = load_svmlight_file("a9a.t", n_features=123)
    x_val = add_x_b(x_val.toarray())
    y_val = numpy.array([int(y_item / 2 + 0.5) for y_item in y_val])
    y_val = y_val.reshape(len(y_val), 1)

    return x_train, y_train, x_val, y_val


# 把训练集添加一列全1，即添加参数b
def add_x_b(x):
    b_train = numpy.ones(x.shape[0])
    return numpy.c_[x, b_train]


# 获取小批量数据
def get_mini_batch(batch_size, x_train, y_train):
    # list of random indices
    lis = random.sample(range(x_train.shape[0]), batch_size)
    x_mb = x_train[lis[0]]
    y_mb = y_train[lis[0]]
    for index in lis[1:]:
        x_mb = numpy.vstack((x_mb, x_train[index]))
        y_mb = numpy.vstack((y_mb, y_train[index]))
    # 特殊情况
    if batch_size == 1:
        x_mb = x_mb.reshape(1, -1)
        y_mb = y_mb.reshape(-1, 1)
    return x_mb, y_mb


def sigmoid(in_x):
    return 1.0 / (1 + numpy.exp(-in_x))


# 计算损失
def compute_lost(x, y, w):
    w = w.reshape(len(w), 1)
    predict = x.dot(w)
    h = sigmoid(predict)
    lost = 0.0
    for i in range(x.shape[0]):
        if y[i] == 1:
            lost -= math.log(h[i, 0])
        else:
            lost -= math.log(1 - h[i, 0])
    return lost / x.shape[0]


# 计算阈值损失
def threshold_lost(x, y, w, th):
    w = w.reshape(len(w), 1)
    predict = x.dot(w)
    h = sigmoid(predict)
    lost = 0.0
    for i in range(x.shape[0]):
        temp = int(h[i, 0] >= th)
        if y[i] == 1:
            lost -= temp * math.log(h[i, 0])
        else:
            lost -= temp * math.log(1 - h[i, 0])
    return lost / x.shape[0]


# 批量随机梯度下降更新方法
def regression_sgd(x_train, y_train, x_val, y_val, w, num_iter, learning_rate, th, batch_size):
    lost_train = list()
    lost_val = list()
    lost_threshold = list()

    # 迭代
    lost_train.append(compute_lost(x_train, y_train, w))
    lost_val.append(compute_lost(x_val, y_val, w))
    lost_threshold.append(threshold_lost(x_val, y_val, w, th))

    for i in range(1, num_iter + 1):
        x_mb, y_mb = get_mini_batch(batch_size, x_train, y_train)
        #  h = sigmoid(W*X+b)
        h = sigmoid(numpy.dot(x_mb, w))
        # x_train_batch.shape[0]  x 的数量
        w_theta = numpy.dot(x_mb.transpose(), h - y_mb) * 1.0 / x_mb.shape[0]
        # 更新参数
        w -= learning_rate * w_theta
        lost_train.append(compute_lost(x_train, y_train, w))
        lost_val.append(compute_lost(x_val, y_val, w))
        lost_threshold.append(threshold_lost(x_val, y_val, w, th))
    return lost_train, lost_val, lost_threshold


x_train, y_train, x_val, y_val = process_data()
# 全0初始化参数矩阵
W = numpy.zeros((x_train.shape[1], 1))

# 随机梯度下降不同学习率的比较
train, val, threshold = regression_sgd(x_train, y_train, x_val, y_val,
                                       W.copy(),
                                       1000,
                                       0.15,
                                       0.5,
                                       64)
train2, val2, threshold2 = regression_sgd(x_train, y_train, x_val, y_val,
                                          W.copy(),
                                          1000,
                                          0.1,
                                          0.5,
                                          64)
train3, val3, threshold3 = regression_sgd(x_train, y_train, x_val, y_val,
                                          W.copy(),
                                          1000,
                                          0.05,
                                          0.5,
                                          64)
plt.figure(1)
plt.xlabel('epoch')
plt.ylabel('lost')
plt.title('lost curve comparing')
plt.plot(val, label='val lr = 0.15')
plt.plot(val2, label='val lr = 0.1')
plt.plot(val3, label='val lr = 0.05')
plt.legend()
plt.show()

plt.figure(2)
plt.xlabel('epoch')
plt.ylabel('lost')
plt.title('lost curve comparing')
plt.plot(train, label='train lr = 0.15')
plt.plot(val, label='val lr = 0.15')
plt.plot(threshold, label='threshold lr = 0.15')
plt.legend()
plt.show()

plt.figure(3)
plt.xlabel('epoch')
plt.ylabel('lost')
plt.title('lost curve comparing')
plt.plot(train2, label='train lr = 0.1')
plt.plot(val2, label='val lr = 0.1')
plt.plot(threshold2, label='threshold lr = 0.1')
plt.legend()
plt.show()

plt.figure(4)
plt.xlabel('epoch')
plt.ylabel('lost')
plt.title('lost curve comparing')
plt.plot(train3, label='train lr = 0.05')
plt.plot(val3, label='val lr = 0.05')
plt.plot(threshold3, label='threshold lr = 0.05')
plt.legend()
plt.show()
