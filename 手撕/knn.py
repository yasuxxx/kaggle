import numpy as np
# knn python实现
class KNearestNeighbor(object):
   def __init__(self):
       pass

   # 训练函数
   def train(self, X, y):
       self.X_train = X
       self.y_train = y

   # 预测函数
   # 需要的numpy_api: np.zeros、np.dot(x_test,x_train)、np.sum(keepdims=True)、np.bincount()
   def predict(self, X, k=1):
       # 计算L2距离
       num_test = X.shape[0]
       num_train = self.X_train.shape[0]
       dists = np.zeros((num_test, num_train))    # 初始化距离函数
       # because(X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train, so
       d1 = -2 * np.dot(X, self.X_train.T)    # shape (num_test, num_train)
       d2 = np.sum(np.square(X), axis=1, keepdims=True)    # shape (num_test, 1)
       d3 = np.sum(np.square(self.X_train), axis=1)    # shape (1, num_train)
       dist = np.sqrt(d1 + d2 + d3)
       # 根据K值，选择最可能属于的类别
       y_pred = np.zeros(num_test)
       for i in range(num_test):
           dist_k_min = np.argsort(dist[i])[:k]    # 最近邻k个实例位置
           y_kclose = self.y_train[dist_k_min]     # 最近邻k个实例对应的标签
           y_pred[i] = np.argmax(np.bincount(y_kclose.tolist()))    # 找出k个标签中从属类别最多的作为预测类别

       return y_pred
