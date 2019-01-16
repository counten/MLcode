# -*- coding:utf-8 -*-
"""
Created on 18-11-24
Author: wbq813 (wbq813@foxmail.com)
Python3.6.6 deepin15.7
"""
import random

import numpy
from matplotlib import pyplot


# 加载文件数据
def load_data():
    data_base = numpy.loadtxt('ml-100k/u1.base', delimiter='\t', usecols=(0, 1, 2), dtype=int)
    data_test = numpy.loadtxt('ml-100k/u1.test', delimiter='\t', usecols=(0, 1, 2), dtype=int)
    return data_base, data_test


# 数据转为矩阵
def data_2_matrix(data, n, m):
    matrix = numpy.zeros((n, m))
    for user_id, item_id, rating in data:
        matrix[user_id - 1][item_id - 1] = int(rating)
    return matrix


def draw(data):
    import numpy as np
    from matplotlib import pyplot as plt
    bins = np.arange(0, 6, 0.1)  # fixed bin size
    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('count on different rating')
    plt.xlabel('rating')
    plt.ylabel('count')
    plt.show()


# 获取小批量数据
def get_mini_batch(batch_size, data_train):
    # list of random indices
    lis = random.sample(range(data_train.shape[0]), batch_size)
    x_mb = data_train[lis[0]]
    for index in lis[1:]:
        x_mb = numpy.vstack((x_mb, data_train[index]))
    # 特殊情况
    if batch_size == 1:
        x_mb = x_mb.reshape(1, -1)
    return x_mb


# RMSE 评估
def RMSE(valid, m_r):
    rmse_temp = list()
    for j in range(len(valid)):
        e2 = (valid[j, 2] - m_r[valid[j, 0], valid[j, 1]]) ** 2
        rmse_temp.append(e2)
    return numpy.sqrt(numpy.mean(rmse_temp))


def matrix_factorization(data_train, data_test, user, item, K, batch_size, epoch, learn_rate,
                         plenty):
    plenty = float(plenty)
    # 测试集填充矩阵
    matrix_test = data_2_matrix(data_test, user, item)
    # 初始化因子矩阵

    m_u = numpy.random.rand(user, K)
    m_i = numpy.random.rand(item, K)
    loss_val = list()
    for step in range(epoch):
        # 随机获取批量数据，并填充矩阵
        train_batch = get_mini_batch(batch_size, data_train)
        matrix_train = data_2_matrix(train_batch, user, item)

        # 实际评分和预测评分差
        m_diff = matrix_train - numpy.dot(m_u, m_i.T)

        for i in range(matrix_train.shape[0]):
            for j in range(matrix_train.shape[1]):
                if matrix_train[i][j] > 0:
                    for k in range(K):
                        # 更新因子矩阵
                        temp = m_u[i][k]
                        m_u[i][k] += learn_rate * (2 * m_diff[i][j] * m_i[j][k] - plenty * temp)
                        m_i[j][k] += learn_rate * (2 * m_diff[i][j] * temp - plenty * m_i[j][k])

        # 计算 loss
        loss = 0.0
        m_diff_test = matrix_test - numpy.dot(m_u, m_i.T)
        for i in range(matrix_test.shape[0]):
            for j in range(matrix_test.shape[1]):
                if matrix_test[i][j] > 0:
                    loss += pow(m_diff_test[i][j], 2)
                    for k in range(K):
                        loss += (plenty / 2) * (pow(m_u[i][k], 2) + pow(m_i[j][k], 2))
        loss_val.append(loss)

        # 提前结束
        if loss < 0.001:
            break
    return m_u, m_i, loss_val


if __name__ == "__main__":
    # 加载数据
    u = 943
    i = 1682
    data_train, data_test = load_data()
    before = data_2_matrix(data_train, u, i)
    draw(before.flatten())

    # SGD 矩阵分解
    nP, nQ, loss_val = matrix_factorization(data_train=data_train.copy(),
                                            data_test=data_test.copy(),
                                            user=u,
                                            item=i,
                                            K=3,
                                            batch_size=10000,
                                            epoch=200,
                                            learn_rate=0.0005,
                                            plenty=0.02)

    predict = numpy.dot(nP, nQ.T)
    draw(predict.flatten())
    nP2, nQ2, loss_val2 = matrix_factorization(data_train=data_train,
                                               data_test=data_test,
                                               user=u,
                                               item=i,
                                               K=6,
                                               batch_size=10000,
                                               epoch=200,
                                               learn_rate=0.0005,
                                               plenty=0.02)
    predict = numpy.dot(nP2, nQ2.T)
    draw(predict.flatten())
    nP3, nQ3, loss_val3 = matrix_factorization(data_train=data_train,
                                               data_test=data_test,
                                               user=u,
                                               item=i,
                                               K=9,
                                               batch_size=10000,
                                               epoch=200,
                                               learn_rate=0.0005,
                                               plenty=0.02)
    predict = numpy.dot(nP3, nQ3.T)
    draw(predict.flatten())
    # 画loss变化图
    pyplot.plot(loss_val, "-", color="b", label="lr=0.005,k=3")
    pyplot.plot(loss_val2, "-", color="r", label="lr=0.005,k=6")
    pyplot.plot(loss_val3, "-", color="y", label="lr=0.005,k=9")
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.legend()
    pyplot.show()

    # RMSE 随K的变化图
    # score = list()
    # for k in range(1, 21):
    #     P, Q, _ = matrix_factorization(data_train=data_train,
    #                                    data_test=data_test,
    #                                    user=943,
    #                                    item=1682,
    #                                    K=k,
    #                                    batch_size=10000,
    #                                    epoch=200,
    #                                    learn_rate=0.0005,
    #                                    plenty=0.02)
    #     score.append(RMSE(data_test, numpy.dot(P, Q.T)))
    #     print(k)
    #
    # pyplot.scatter(range(1, 51), score)
    # pyplot.suptitle('Size of RMSE by number of latent factors')
    # pyplot.xlabel('number of factors')
    # pyplot.ylabel('RMSE')
    # pyplot.show()
