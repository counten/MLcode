# -*- coding:utf-8 -*-
"""
Created on 18-12-9
Author: wbq813 (wbq813@foxmail.com)
Python3.6.6 deepin15.7
"""

import matplotlib.pyplot as plt
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


# 计算损失
def compute_lost(X, y, W, reg):
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    # 对于每一个样本，累加loss
    for i in range(num_train):
        # 计算出x下标i的类别向量
        scores = X[i].dot(W)  # (1, C)
        # 得到x下标i正确的类别向量，y[i]代表x下标i个数据是第y[i]个类
        # y[i]属于[0....类别的个数C]
        correct_class_score = scores[int(y[i][0])]
        # 遍历每个类别，计算lost
        for j in range(num_classes):
            if j == y[i][0]:
                continue
            # delta = 1
            margin = scores[j] - correct_class_score + 1
            # max(0, yi - yc + 1)
            if margin > 0:
                loss += margin
    # 训练数据平均损失
    loss /= num_train
    # 正则损失
    loss += reg * numpy.sum(W * W)
    return loss


# 计算梯度
def svm_gradient(W, X, y, reg):
    dW = numpy.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        # 计算出x下标i的类别向量
        scores = X[i].dot(W)  # (1, C)
        # 得到x下标i正确的类别向量，y[i]代表x下标i个数据是第y[i]个类
        # y[i]属于[0....类别的个数C]
        correct_class_score = scores[int(y[i][0])]
        # 遍历每个类别，计算lost
        for j in range(num_classes):
            if j == y[i][0]:
                continue
            # 假设delta = 1
            margin = scores[j] - correct_class_score + 1
            # max(0, yi - yc + 1)
            if margin > 0:
                dW[:, y[i][0]] += -X[i, :]
                dW[:, j] += X[i, :]
    # 训练数据平均损失
    dW /= num_train
    # 加上正则的偏导
    dW += 2 * reg * W
    return dW


def classify_msgd(X_train, y_train, X_test, y_test, W, epoch, learn_rate, reg, batch_size):
    loss_train = list()
    loss_val = list()
    for i in range(epoch):
        x_mb, y_mb = get_mini_batch(batch_size, X_train, y_train)
        grad = svm_gradient(W, x_mb, y_mb, reg)
        W -= learn_rate * grad
        loss_train.append(compute_lost(X_train, y_train, W, reg))
        loss_val.append(compute_lost(X_test, y_test, W, reg))
    return loss_train, loss_val


X_train, y_train, X_test, y_test = process_data()
# 对权重进行初始化
W = numpy.random.randn(X_train.shape[1], 2) * 0.01

plt.figure(1)
plt.xlabel('epoch')
plt.ylabel('lost')
plt.title('lost curve comparing')
train, val = classify_msgd(X_train, y_train, X_test, y_test, W.copy(),
                           epoch=500,
                           learn_rate=0.1,
                           reg=0.01,
                           batch_size=64)
train2, val2 = classify_msgd(X_train, y_train, X_test, y_test, W.copy(),
                             epoch=500,
                             learn_rate=0.01,
                             reg=0.01,
                             batch_size=64)
train3, val3 = classify_msgd(X_train, y_train, X_test, y_test, W.copy(),
                             epoch=500,
                             learn_rate=0.001,
                             reg=0.01,
                             batch_size=64)
plt.figure(1)
plt.xlabel('epoch')
plt.ylabel('lost')
plt.title('lost curve comparing')
plt.plot(val, label='val lr = 0.1')
plt.plot(val2, label='val lr = 0.01')
plt.plot(val3, label='val lr = 0.001')
plt.legend()
plt.show()

plt.figure(2)
plt.xlabel('epoch')
plt.ylabel('lost')
plt.title('lost curve comparing')
plt.plot(train, label='train lr = 0.1')
plt.plot(val, label='val lr = 0.1')
plt.legend()
plt.show()

plt.figure(3)
plt.xlabel('epoch')
plt.ylabel('lost')
plt.title('lost curve comparing')
plt.plot(train2, label='train lr = 0.01')
plt.plot(val2, label='val lr = 0.01')
plt.legend()
plt.show()

plt.figure(4)
plt.xlabel('epoch')
plt.ylabel('lost')
plt.title('lost curve comparing')
plt.plot(train3, label='train lr = 0.001')
plt.plot(val3, label='val lr = 0.001')
plt.legend()
plt.show()
