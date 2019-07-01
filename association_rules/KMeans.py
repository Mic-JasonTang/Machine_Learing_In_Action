#!/user/bin/python
# -*- coding:utf -8 -*-

import numpy as np
import matplotlib.pyplot as plt

#K-均值聚类支持函数
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return np.mat(dataMat)

def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * np.random.rand(k,1)
    return centroids

#K-均值聚类算法
def kMeans2(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:1], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
                clusterAssment[i,:] = minIndex, minDist ** 2
        # print centroids
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment

# k-means算法，distMeas为计算距离的函数，creatCent为生成质心的函数
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))  # 簇结果分配矩阵，记录簇索引值和存储误差（当前点到簇质心的距离）
    centroids = createCent(dataSet, k)  # 生成k组（k*2)质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 按行遍历所有数据
            minDist = np.inf;
            minIndex = -1  # 定义最小距离为无穷大
            for j in range(k):  # 寻找最近的质心
                distJI = distMeas(centroids[j, :], dataSet[i, :])  # 计算当前组质心与第i行数据的距离
                if distJI < minDist:  # 符合要求就记录距离，记录簇索引值
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:  # 当前分配的簇应该是簇最小索引，如果不是，则继续分配
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2  # 记录最小质心的下标，距离
        # print centroids
        for cent in range(k):  # 更新质心的位置
            # nonzero 返回满足条件的下标，返回的是一个行、列两个array。
            # .A 返回的是一个numpy数组
            # clusterAssment[:, 0] 存放的是最小质心的下标（0-k）。
            # np.nonzero(clusterAssment[:, 0].A == cent) 返回质心相等的那些下标
            # np.nonzero(clusterAssment[:, 0].A == cent)[0] 返回质心相等的那些下标所在的行，不要列
            # print "质心相等小标所在行为: ", np.nonzero(clusterAssment[:, 0].A == cent)[0]
            # 取出输入cent簇的那些数据出入ptsInClust
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 把这一簇的均值更新为质心
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def showCluster(dataSet, k, clusterAssment, centroids):
    print(dataSet.shape)
    fig = plt.figure()
    plt.title("K-means")
    ax = fig.add_subplot(111)
    data = []
    for cent in range(k): #提取出每个簇的数据
        ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]] #获得属于cent簇的数据
        data.append(ptsInClust)
    for cent, c, marker in zip( range(k), ['r', 'g', 'b', 'y'], ['^', 'o', '*', 's'] ): #画出数据点散点图
        print(data[cent][:, 0], data[cent][:, 1])
        ax.scatter(data[cent][:, 0].tolist(), data[cent][:, 1].tolist(), s=80, c=c, marker=marker)
    ax.scatter(centroids[:, 0].tolist(), centroids[:, 1].tolist(), s=1000, c='black', marker='+', alpha=1) #画出质心点
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.show()


dataSet = loadDataSet('k-means_data/testSet.txt')
datMat = np.mat(dataSet)
# print '最小值:'
# print min(datMat[:,0])
# print min(datMat[:,1])
#
# print '最大值：'
# print max(datMat[:,0])
# print max(datMat[:,1])
#
# print '距离：'
# print distEclud(datMat[0], datMat[1])

myCentroids, clustAssing = kMeans2(datMat, 4)
# print myCentroids
# print clustAssing

# print '如图所示:'
# plt.show()

showCluster(dataSet, 4, clustAssing, myCentroids)