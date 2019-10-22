# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random
import math
import time

def loadData(mu0,sigma0,mu1,sigma1,alpha0,alpha1):
    data_size = 10000

    data0 = np.random.normal(mu0, sigma0, int(data_size * alpha0))
    data1 = np.random.normal(mu1, sigma1, int(data_size * alpha1))

    data = []
    data.extend(data0)
    data.extend(data1)

    return data

def calcGauss(dataSetArr, mu, sigmod):
    return norm.pdf(dataSetArr,mu,sigmod)

def E_step(data,mu0,sigma0,mu1,sigma1,alpha0,alpha1):
    """
    计算的Q函数
    :return:
    """
    gamma_0 = alpha0 * calcGauss(data, mu0, sigma0)
    gamma_1 = alpha1 * calcGauss(data, mu1, sigma1)

    sum = gamma_0 + gamma_1

    gamma_0 = gamma_0 / sum
    gamma_1 = gamma_1 / sum

    return gamma_0,gamma_1

def M_step(mu0, mu1, gamma0, gamma1, data):
    mu0_new = np.dot(gamma0, data) / np.sum(gamma0)
    mu1_new = np.dot(gamma1, data) / np.sum(gamma1)

    sigma0 = np.sqrt(np.dot(gamma0, np.square(data - mu0)) / np.sum(gamma0))
    sigma1 = np.sqrt(np.dot(gamma1, np.square(data - mu1)) / np.sum(gamma1))

    alpha0 = np.sum(gamma0) / len(gamma0)
    alpha1 = np.sum(gamma1) / len(gamma1)

    return mu0_new,mu1_new,sigma0,sigma1,alpha0,alpha1

def EM(data):
    data = np.array(data)
    alpha0 = 0.5
    mu0 = 0
    sigma0 = 1 # Standard deviation
    alpha1 = 0.5
    mu1 = 1
    sigma1 = 1 # Standard deviation

    step = 0
    while step < 1000:
        gamma0, gamma1 = E_step(data, mu0, sigma0, mu1, sigma1, alpha0, alpha1)
        mu0,mu1,sigma0,sigma1,alpha0,alpha1 = M_step(mu0, mu1, gamma0, gamma1, data)
        step += 1

    print("mu0: {:.8f}  mu1: {:.8f} \n"
          "sigma0: {:.8f}  simga1:{:.8f}\n"
          "alpha0:{:.8f}  alpha1:{:.8f}.\n".
          format(mu0,mu1,sigma0,sigma1,alpha0,alpha1))



if __name__ == '__main__':
    data = loadData(mu0=1,sigma0=2,mu1=10,sigma1=1,alpha0=0.3,alpha1=0.7)
    # plt.hist(data,bins=50)
    # plt.show()
    mu0_ = 1
    sigma0_ = 2
    mu1_ = 10
    sigma1_ = 1
    alpha0_ = 0.3
    alpha1_ = 0.7
    print("mu0: {:.2f}  mu1: {:.2f} \n"
          "sigma0: {:.2f}  simga1:{:.2f}\n"
          "alpha0:{:.2f}  alpha1:{:.2f}.".
          format(mu0_,mu1_,sigma0_,sigma1_,alpha0_,alpha1_))
    print("=" * 20)
    EM(data)
