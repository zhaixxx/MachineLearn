from Demo_KNN.model_selection import train_test_split
from sklearn import datasets
from Demo_KNN.KNNmode import KNNClassifier

iris = datasets.load_iris()

X = iris.data   # 数据集
y = iris.target  # 数据集对应特征

X_train,X_test,y_train,y_test = train_test_split(X,y)
# print(X_test.shape,y_train.shape)
my_knn_clf = KNNClassifier(k=3)

my_knn_clf.fit(X_train,y_train)

y_predict = my_knn_clf.predict(X_test)

# print(y_predict)

value =sum(y_predict == y_test)/len(y_test)
print(value)



