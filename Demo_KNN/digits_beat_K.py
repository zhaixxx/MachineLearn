import  numpy as np
from sklearn import datasets
from Demo_KNN.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

digits = datasets.load_digits()
x = digits.data
y = digits.target
x_train,x_test,y_tarin,y_test = train_test_split(x,y,test_ratio=0.2,seed = 666)
# knn_clf = KNeighborsClassifier(n_neighbors=3)
# knn_clf.fit(x_train,y_tarin)
# score = knn_clf.score(x_test,y_test)
# print(score)
best_score = 0.0
best_k = -1
best_p = -1
# 网格搜索策略
# for k in range(1,11):
#     for p in range(1,6):
#         knn_clf = KNeighborsClassifier(n_neighbors=k,weights="distance",p=p)
#         knn_clf.fit(x_train,y_tarin)
#         score = knn_clf.score(x_test,y_test)
#         if score>best_score:
#             best_k = k
#             best_score = score
#             best_p = p
# print(best_score,best_k,best_p)
#
knn_clf = KNeighborsClassifier()
praram_grid = [
    {
        'weights':['uniform'],
        'n_neighbors':[i for i in range (1,11)]
    },
    {
        'weights':['distance'],
        # 设置 KNN距离计算公式
        'n_neighbors':[i for i in range (1,11)],
        # K的大小
        'p':[i for i in range(1,6)]
        # 幂的大小
    }
]
grid_search = GridSearchCV(knn_clf,praram_grid, n_jobs=-1, verbose=2)
# 分类器，自定义标准，最大核心运行,运行进度
grid_search.fit(x_train,y_tarin)
print(grid_search.best_estimator_)

knn_clf = grid_search.best_estimator_
# 得到最佳分类器
