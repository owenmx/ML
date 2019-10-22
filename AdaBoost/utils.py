# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

def KL_divergence(p,q):
    sum = 0.0
    for p_,q_ in zip(p,q):
        sum += p_ * np.log(p_ / q_)
    return sum

if __name__ == '__main__':
    p = [0.2,0.8]
    q = [0.9,0.1]

    plt.figure(1)
    plt.bar([1,2],height=p)
    plt.figure(2)
    plt.bar([1,2],height=q)

    plt.show()
    print(KL_divergence(p,q))



