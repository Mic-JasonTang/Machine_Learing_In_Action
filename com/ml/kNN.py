#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/19 22:15
# @Author  : Jasontang
# @Site    : 
# @File    : kNN.py
# @ToDo    : K-近邻算法


import numpy as np
import operator


def createDataSet():
	group = np.array([[1.0, 1.1],
				   [1.0, 1.0],
				   [0, 0],
				   [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0 (inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]

	# ---- 开始计算欧式距离 -----
	# print np.tile(inX, (dataSetSize, 1))
	diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
	# print diffMat
	sqDiffMat = diffMat ** 2
	# print sqDiffMat
	# print sqDiffMat.sum()
	# print sqDiffMat.sum(axis=1)
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances ** 0.5
	# ---- 计算结束 ------
	print distances
	# 进行排序并返回索引
	sortedDistIndicies = distances.argsort()
	print sortedDistIndicies
	classCount = {}

	# 选择距离最小的k个点
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	# 排序
	for item in classCount.iteritems():
		print item

	sortedClassCount = sorted(classCount.iteritems(),
							  key = operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]
