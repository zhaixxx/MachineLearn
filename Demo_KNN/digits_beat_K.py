import  numpy as np
from sklearn import datasets
from Demo_KNN.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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
for k in range(1,11):
    for p in range(1,6):
        knn_clf = KNeighborsClassifier(n_neighbors=k,weights="distance",p=p)
        knn_clf.fit(x_train,y_tarin)
        score = knn_clf.score(x_test,y_test)
        if score>best_score:
            best_k = k
            best_score = score
            best_p = p
print(best_score,best_k,best_p)