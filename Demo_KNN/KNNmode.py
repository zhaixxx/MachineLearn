import numpy as np
from math import sqrt
from collections import Counter
from Demo_KNN.metrics import accuracy_score
class KNNClassifier:

    def __init__(self,k):
        assert k >=1,\
        "k must be valid"

        self.k = k
        self._X_train = None
        self._y_tarin = None

    def fit(self,X_train , y_train):
        # "根据训练数据集X_tran和y_train训练KNN分类器"
        self._X_train = X_train
        self._y_tarin = y_train
        return self

    def predict(self,X_predict):
        assert self._X_train is not None and self._y_tarin is not None,\
        "must fit before predict"
        assert X_predict.shape[1] == self._X_train.shape[1],\
        "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        assert x.shape[0] == self._X_train.shape[1],\
        "the feature number of x must be eaual to X_train"

        diistances = [sqrt(np.sum((x_train-x) **2)) for x_train in self._X_train]
        nearset = np.argsort(diistances)

        topK_y = [self._y_tarin[i] for i in nearset[:self.k]]
        votes = Counter(topK_y)  # 取出投票结果前K个值
        return votes.most_common(1)[0][0]

    def socre(self,x_test,y_test):
        # "根据测试数据集 X_test 和 y_test 确定当前模型的准确度"

        y_predict = self.predict(x_test)
        return accuracy_score(y_test,y_predict)

    def __repr__(self):
        return "KNN(k=%d)"% self.k
