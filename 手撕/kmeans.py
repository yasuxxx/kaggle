# -*- coding:utf-8 -*-
import numpy as np

# 实现kmeans
# 需要使用的函数：计算距离 np.linalg.norm()
# 找到最小距离对应的索引：distances.index(min(distances))
# 计算某个轴上的平均值：np.average(matricx,axis=0)
class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, k=2, tolerance=0.0001, max_iter=300):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        self.centers_ = {}
        # 随机选择k个点作为中心点
        for i in range(self.k_):
            self.centers_[i] = data[i]

        for i in range(self.max_iter_):
            # 分类簇
            self.clf_ = {}
            for i in range(self.k_):
                self.clf_[i] = []
            for feature in data:
                distances = []
                # 当前feature点到k个中心点的距离
                for center in self.centers_:
                    distances.append(np.linalg.norm(feature - self.centers_[center]))
                # 找到当前feature点所属的那个类
                classification = distances.index(min(distances))
                # 装进所属那个类的类簇中
                self.clf_[classification].append(feature)

            prev_centers = dict(self.centers_)
            # 重新计算k个中心点的位置
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c], axis=0)

            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False
            if optimized:
                break

    def predict(self, p_data):
        # 预测时先算到k个聚类中心的距离
        # 输出最小的一个聚类中心即可
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index