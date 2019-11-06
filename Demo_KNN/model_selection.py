import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


def train_test_split(X,y,test_ratio=0.2,seed=None):
    assert X.shape[0] == y.shape[0],\
    "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0,\
    "test_ration must be valid"

    if seed:
        np.random.seed(seed)  # 随机种子

    #  因为数据集非常规则，因此需要对数据集和特征值进行随机化
    shuffle_indexes = np.random.permutation(len(X))  # 对0-（len(x)-1）个数字进行乱序排序
    # test_ratio = 0.2  # 测试数据集
    test_size = int(len(X) * test_ratio)
    # print(test_size)
    test_indexes = shuffle_indexes[:test_size]  # 测试数据
    train_indexes = shuffle_indexes[test_size:]  # 训练数据
    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    # print(y_test.shape,X_test.shape)
    return X_train,X_test,y_train,y_test
