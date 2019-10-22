import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def KL_divergence(p,q):
    counter_p = Counter(p)
    counter_q = Counter(q)

    pro_p = list()
    pro_q = list()

    val_p = np.array(list(counter_p.values()))
    val_q = np.array(list(counter_q.values()))

    for k in counter_p.keys():
        pro_p.append(counter_p[k] / np.sum(val_p))
        pro_q.append(counter_q[k] / np.sum(val_q))

    pro_p = np.array(pro_p)
    pro_q = np.array(pro_q)

    print(pro_p)
    print(pro_q)
    pro_p = [0.4, 0.6]
    pro_q = [0.6, 0.4]
    kl = 0.0
    for p_,q_ in zip(pro_p,pro_q):
        kl += p_ * np.log(p_ / q_)
    print(kl)

def load_data():
    data = pd.read_csv("dataset/agaricus-lepiota.csv")
    y = data['target']
    X = data.drop('target',axis=1)

    # split data
    stalks = X['A10'].values
    index_d = np.where(stalks == 't')[0]
    index_s = np.where(stalks == 'e')[0]

    for col in X.columns:
        unique = list(set(X[col].values.tolist()))
        map = dict(zip(unique,[i for i in range(len(unique))]))
        X[col] = X[col].map(map)
    y = y.map({"p":0,"e":1})

    Td_X = X.iloc[index_d]
    Td_Y = y.iloc[index_d]

    Ts_S_X = X.iloc[index_s]
    Ts_S_Y = y[index_s]

    data_d_x = Td_X.values
    data_d_y = Td_Y.values

    data_s_x = Ts_S_X.values
    data_s_y = Ts_S_Y.values

    data_s_X_train, data_s_X_test, \
    data_s_y_train, data_s_y_test = \
        train_test_split(data_s_x,data_s_y,test_size=0.33,random_state=42)

    return data_d_x,data_d_y,data_s_X_train,data_s_y_train,data_s_X_test,data_s_y_test

def normalize(data):
    sum_data = np.sum(data)
    data = data / sum_data
    return data

def tradaboost(A_X,A_y,B_X_train,B_y_train):
    d_raw_num,d_feature_num = A_X.shape
    s_raw_num, s_feature_num = B_X_train.shape

    union_data_X = np.concatenate([A_X, B_X_train],axis=0)
    union_data_Y = np.concatenate([A_y, B_y_train], axis=0)

    w = np.array([i / (d_raw_num + s_raw_num) for i in range(d_raw_num + s_raw_num)])
    base_learner = DecisionTreeClassifier(max_depth=3)
    train_time = 20

    train = 0
    while train < train_time:
        w = normalize(w)

        base_learner.fit(union_data_X, union_data_Y, sample_weight=w)

        y_pred = base_learner.predict(union_data_X)

        err = np.sum(w * (y_pred != union_data_Y))

        alpha = err / (1 - err)
        beta = 1 / (1 + np.sqrt(2 * np.log(d_raw_num) / train_time))

        # different
        w[:d_raw_num] = w[:d_raw_num] * np.power(alpha, (y_pred[:d_raw_num] != union_data_Y[:d_raw_num]))

        # same
        w[d_raw_num:] = w[d_raw_num:] * np.power(alpha, -(y_pred[:d_raw_num] != union_data_Y[:d_raw_num]))

        train += 1
    pass

if __name__ == '__main__':
    A_X, A_y, B_X_train, B_y_train, B_X_test, B_y_test = load_data()


    clf = DecisionTreeClassifier()
    clf.fit(A_X,A_y)

    print(len(B_X_train))

    print("acc:",accuracy_score(clf.predict(B_X_test),B_y_test))
    pass
