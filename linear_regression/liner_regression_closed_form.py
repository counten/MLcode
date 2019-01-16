# -*- coding:utf-8 -*-
"""
Created on 18-12-9
Author: wbq813 (wbq813@foxmail.com)
Python3.6.6 deepin15.7
"""
import numpy
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split


def process_data(test_size):
    x, y = load_svmlight_file("housing_scale", n_features=13)
    x = x.toarray()
    # 增加一列，后面w对应增加一列作为参数b
    # x = numpy.column_stack((x, numpy.ones((x.shape[0], 1))))
    y = y.reshape((-1, 1))

    # 划分测试集和验证集
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size)
    return x_train, x_val, y_train, y_val


def liner_regression_closed_form(x_train, x_val, y_train, y_val, lamd):
    # 计算参数
    denominator = numpy.matrix(numpy.dot(x_train.transpose(), x_train) + lamd).I
    molecule = numpy.dot(x_train.transpose(), y_train)
    w = numpy.dot(denominator, molecule)
    print(w)
    # 计算训练集下的损失
    loss_train = numpy.average(numpy.abs(numpy.dot(x_train, w) - y_train))

    # 计算验证集下的损失
    loss_val = numpy.average(numpy.abs(numpy.dot(x_val, w) - y_val))

    return loss_train, loss_val


if __name__ == '__main__':
    test_size = 0.25
    x_train, x_val, y_train, y_val = process_data(test_size)
    loss_train, loss_val = liner_regression_closed_form(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        lamd=0
    )
    print("loss of train set: %s" % loss_train)
    print("loss of valid set: %s" % loss_val)
