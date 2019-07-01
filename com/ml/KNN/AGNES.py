#!/user/bin/python
# -*- coding: utf -8 -*-

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        # curLine = line.strip().split('\n')
        curLine = line.strip().split(' ')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return mat(dataMat)

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

#dist_min
def dist_min(Ci, Cj):
    return min(distEclud(i, j) for i in Ci for j in Cj)

#dist_max
def dist_max(Ci, Cj):
    return max(distEclud(i, j) for i in Ci for j in Cj)

#dist_avg
def dist_avg(Ci, Cj):
    return sum(distEclud(i, j) for i in Ci for j in Cj)/(len(Ci) * len(Cj))

#找到距离最小的下标
def find_Min(M):
    min = 1000
    x = 0
    y = 0
    for i in range(len(M)):
        for j in range(len(M[i])):
            if i != j and M[i][j] < min:
                min = M[i][j]
                x = i
                y = j
    return (x, y, min)

#AGNES算法
def AGNES(dataSet, distEclud, k):
    #初始化C和M
    C = []
    M = []
    for i in dataSet:
        Ci = []
        Ci.append(i)
        C.append(Ci)
    for i in C:
        Mi = []
        for j in C:
            Mi.append(distEclud(i, j))
        M.append(Mi)
    q = len(dataSet)
    while q > k:
        x, y, min = find_Min(M)
        C[x].extend(C[y])
        print(C)
        print(C[y])
        C.remove(C[y])
        M = []
        for i in C:
            Mi = []
            for j in C:
                Mi.append(distEclud(i, j))
                M.append(Mi)
        q -= 1
    return C

#画图
def draw(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(C)):
        coo_X = []    #x坐标列表
        coo_Y = []    #y坐标列表
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        plt.scatter(coo_X, coo_Y, marker='x', color=colValue[i%len(colValue)], label=i)

    plt.legend(loc='upper right')
    plt.title('AGNES')
    plt.show()


# datMat = loadDataSet('xigua.txt')
#
# C = AGNES(datMat, dist_avg, 3)
# draw(C)


