# -*- coding:utf-8 -*-
"""
Created on 18-12-9
Author: wbq813 (wbq813@foxmail.com)
Python3.6.6 deepin15.7
"""

import random
import numpy
from matplotlib import pyplot
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split


# 处理数据
def process_data(test_size):
    x, y = load_svmlight_file("housing_scale", n_features=13)
    x = x.toarray()
    # 增加一列，后面w对应增加一列作为参数b
    # x = numpy.column_stack((x, numpy.ones((x.shape[0], 1))))
    y = y.reshape((-1, 1))

    # 划分测试集和验证集
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size)
    return x_train, x_val, y_train, y_val


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


# 全样本梯度下降
def linear_regression_fgd(x_train, x_val, y_train, y_val, max_epoch, learning_rate, penalty):
    # 初始化参数
    w = numpy.zeros((x_train.shape[1], 1))
    # w = numpy.random.random((n_features + 1, 1))
    # w = numpy.random.normal(1, 1, size=(n_features + 1, 1))

    list_lose_train = list()
    list_lose_val = list()

    for epoch in range(max_epoch):
        # 计算梯度
        diff = numpy.dot(x_train, w) - y_train
        gradient = penalty * w + numpy.dot(x_train.transpose(), diff)

        descent = -gradient
        # 更新参数
        w += learning_rate * descent

        # 计算训练集下的损失
        loss_train = numpy.average(numpy.abs(numpy.dot(x_train, w) - y_train))
        list_lose_train.append(loss_train)

        # 计算验证集上的损失
        loss_val = numpy.average(numpy.abs(numpy.dot(x_val, w) - y_val))
        list_lose_val.append(loss_val)
    return list_lose_train, list_lose_val


# 批量随机梯度下降
def linear_regression_sgd(x_train, x_val, y_train, y_val, batch_size, max_epoch, learning_rate, penalty):
    # 初始化参数
    w = numpy.zeros((x_train.shape[1], 1))
    # w = numpy.random.random((n_features + 1, 1))
    # w = numpy.random.normal(1, 1, size=(n_features + 1, 1))

    list_lose_train = list()
    list_lose_val = list()

    for epoch in range(max_epoch):
        # 随机获取批量数据，计算梯度
        x_mb, y_mb = get_mini_batch(batch_size, x_train, y_train)
        diff = numpy.dot(x_mb, w) - y_mb
        gradient = penalty * w + numpy.dot(x_mb.transpose(), diff)

        descent = -gradient
        # 更新参数
        w += learning_rate * descent

        # 计算训练集下的损失
        loss_train = numpy.average(numpy.abs(numpy.dot(x_train, w) - y_train))
        list_lose_train.append(loss_train)

        # 计算验证集上的损失
        loss_val = numpy.average(numpy.abs(numpy.dot(x_val, w) - y_val))
        list_lose_val.append(loss_val)
    return list_lose_train, list_lose_val


if __name__ == '__main__':
    test_size = 0.25
    x_train, x_val, y_train, y_val = process_data(test_size)

    losses_train_fgd, losses_val_fgd = linear_regression_fgd(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        max_epoch=2000,
        learning_rate=0.0002,
        penalty=0.5
    )
    # print(losses_train_fgd)
    # print(losses_val_fgd)
    losses_train_sgd, losses_val_sgd = linear_regression_sgd(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        batch_size=200,
        max_epoch=2000,
        learning_rate=0.0002,
        penalty=0.5
    )
    # print(losses_train_sgd)
    # print(losses_val_sgd)
    pyplot.plot(losses_train_fgd, "-", label="fgd train loss")
    pyplot.plot(losses_val_fgd, "-", label="fgd valid loss")
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.legend()
    pyplot.title("Loss varying with iterations.")
    pyplot.show()
    _, losses_val_fgd2 = linear_regression_fgd(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        max_epoch=200,
        learning_rate=0.001,
        penalty=0.5
    )
    _, losses_val_fgd3 = linear_regression_fgd(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        max_epoch=200,
        learning_rate=0.00001,
        penalty=0.5
    )
    pyplot.plot(losses_val_fgd2, "-", label="learning_rate=0.001")
    pyplot.plot(losses_val_fgd, "-", label="learning_rate=0.0002")
    pyplot.plot(losses_val_fgd3, "-", label="learning_rate=0.0001")
    pyplot.xlabel("epoch")
    pyplot.ylabel("Valid loss")
    pyplot.legend()
    pyplot.title("Loss varying with iterations.")
    pyplot.savefig("lr_fgd_lr.png")
    pyplot.show()
    pyplot.plot(losses_train_sgd, "-", label="sgd train loss")
    pyplot.plot(losses_val_sgd, "-", label="sgd valid loss")
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.legend()
    pyplot.title("Loss varying with iterations.")
    pyplot.show()

    _, losses_val_sgd2 = linear_regression_sgd(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        batch_size=200,
        max_epoch=200,
        learning_rate=0.001,
        penalty=0.5
    )
    _, losses_val_sgd3 = linear_regression_sgd(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        batch_size=200,
        max_epoch=200,
        learning_rate=0.0001,
        penalty=0.5
    )

    pyplot.plot(losses_val_sgd2, "-", label="learning_rate=0.001")
    pyplot.plot(losses_val_sgd, "-", label="learning_rate=0.0002")
    pyplot.plot(losses_val_sgd3, "-", label="learning_rate=0.0001")
    pyplot.xlabel("epoch")
    pyplot.ylabel("Valid loss")
    pyplot.legend()
    pyplot.title("Loss varying with iterations.")
    pyplot.show()
