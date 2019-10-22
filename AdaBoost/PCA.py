import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def PCA(data:np.array,k):
    m, n = data.shape

    # max_ = np.tile(np.max(data, axis=0), (m, 1))
    # min_ = np.tile(np.min(data, axis=0), (m, 1))
    # data = (data - min_) / (max_ - min_)

    X = data
    Cov_X = np.cov(X.transpose())

    value, vector = np.linalg.eig(Cov_X)

    index = np.argsort(-value)
    select_vector = vector.T[index[:k]]
    pca_data = np.dot(X,select_vector.T)

    # print(data)
    return pca_data


if __name__ == '__main__':
    k = 2
    data = pd.read_csv("dataset/uci-credit-card.csv")
    label = data['target'].values
    data = data.drop("target", axis=1)
    data = data.values

    mean = np.mean(data, axis=0)  # nor
    data = data - np.tile(mean, (data.shape[0], 1))

    # pca_data = PCA(data,k)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(data)

    unique_label = set(label)
    fig = plt.figure()
    # for l in unique_label:
    #     index = np.where(label == l)
    #     plt.scatter(pca_data[index, 0], pca_data[index, 1], marker='x')
    # plt.show()

    print(label)
    ax = Axes3D(fig)
    for l in unique_label:
        index = np.where(label == l)
        plt.scatter(pca_data[index, 0], pca_data[index, 1],
                    pca_data[index, 2], marker='x')

    plt.show()