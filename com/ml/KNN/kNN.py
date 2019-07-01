#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/19 22:15
# @Author  : Jasontang
# @Site    : 
# @File    : kNN.py
# @ToDo    : K-近邻算法


import numpy as np
import operator, os

# 创建数据集
def createDataSet():
	group = np.array([[1.0, 1.1],
				   [1.0, 1.0],
				   [0, 0],
				   [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

# 分类算法
def classify0 (inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	# ---- 开始计算输入点与每一个样本点的欧式距离 -----
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
	# print distances
	# 进行排序并返回索引
	sortedDistIndicies = distances.argsort()
	# print sortedDistIndicies
	classCount = {}

	# 选择距离最小的k个点
	for i in range(k):
		# print labels[sortedDistIndicies[i]]
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	# 排序
	# for item in classCount.iteritems():
	# 	print item

	sortedClassCount = sorted(classCount.items(),
							  key = operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

# 将数据从文件读入到矩阵中
def file2matrix(filename):
	fr = open(filename, 'r')
	# 得到文件的行数
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	# 零填充矩阵
	returnMat = np.zeros((numberOfLines, 3))
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index, :] = listFromLine[0: 3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat, classLabelVector

# 归一化特征值
def autoNorm(dataSet):
	# print dataSet.min()
	# print dataSet.max()
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	# print minVals, maxVals
	ranges = maxVals - minVals
	# normDataSet = np.zeros(np.shape(dataSet))
	# print np.shape(dataSet)
	m = dataSet.shape[0]
	normDataSet = dataSet - np.tile(minVals, (m, 1))
	# print "--------", type(normDataSet)
	# print "+++++", type(np.tile(ranges, (m, 1)))
	normDataSet /= np.tile(ranges, (m, 1))
	return normDataSet, ranges, minVals

def datingClassTest(filename):
	hoRatio = 0.20 # 10%的测试数据
	datingDataMat, datingLabels = file2matrix(filename)
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVects = int(m * hoRatio)
	errorCount = 0.0

	for i in range(numTestVects):
		# 前10%测试，后90%训练
		classifyResult = classify0(normMat[i, :], normMat[numTestVects:m, :],
								   datingLabels[numTestVects:m], 3)
		print('the classifier came back with: %d, the real answer is: %d' % \
			  (classifyResult, datingLabels[i]))
		if classifyResult != datingLabels[i]:
			errorCount += 1.0
	print ("the total(%d) error rate is: %0.2f%%" % (numTestVects, errorCount/float(numTestVects)*100.0))

def classifyPerson():
	resultList = ['not at all', 'in small doses', 'in large doses']
	percentTats = float(input("percentage of time spent playing video games?"))
	ffMiles = float(input("frequent flier miles earned per year?"))
	iceCream = float(input("liters of ice cream consumed per year?"))

	datingDataMat, datingLabels = file2matrix('../datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	print(ranges)
	print(minVals)
	inArr = np.array([ffMiles, percentTats, iceCream])
	classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
	print('You will probably like this person:', resultList[classifierResult - 1])


def img2Vector(filename):
	returnVect = np.zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32*i + j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	hwLabels = []
	# 获取目录的内容
	trainingFileList = os.listdir("trainingDigits")
	m = len(trainingFileList)
	trainingMat = np.zeros((m, 1024))
	for i in range(m):
		# 这里获取的是文件名，如0_0.txt
		fileNameStr = trainingFileList[i]
		# 去掉.txt
		fileStr = fileNameStr.split(".")[0]
		# 去掉_0,提取文件名第一个数字
		classNumStr = int(fileStr.split("_")[0])
		hwLabels.append(classNumStr)
		trainingMat[i, :] = img2Vector('trainingDigits/%s' % fileNameStr)

	testFileList = os.listdir("testDigits")
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split(".")[0]
		classNumStr = int(fileStr.split("_")[0])
		vectorUnderTest = img2Vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print("the classifier came back with: %d, the real answer is : %d" % (classifierResult, classNumStr))
		if classifierResult != classNumStr:
			errorCount += 1.0
	print("\nthe total(%d) number of errors is: %d" % (mTest, errorCount))
	print("\nthe total(%d) error rate is: %0.2f%%" % (mTest, errorCount/float(mTest)*100.0))
