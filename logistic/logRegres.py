#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/25 23:53
# @Author  : Jasontang
# @Site    : 
# @File    : logRegres.py
# @ToDo    : Logistic回归

import numpy as np
import matplotlib.pyplot as plt

# 加载数据集
def loadDataSet():
	dataMat = []; labelMat = []
	x1 = []; y1 = []
	x2 = []; y2 = []
	fr = open("testSet.txt")
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		if int(lineArr[2]) == 1:
			x1.append(float(lineArr[0])); y1.append(float(lineArr[1]))
		else:
			x2.append(float(lineArr[0])); y2.append(float(lineArr[1]))
		labelMat.append(int(lineArr[2]))
	plt.scatter(x1, y1, s=60, c="blue", marker="o", label="Class 1")
	plt.scatter(x2, y2, s=20, c="red", marker="o", label="Class 0")
	plt.xlabel("X1")
	plt.ylabel("X2")
	plt.legend(loc='upper left')
	plt.show()
	return dataMat, labelMat

# sigmod 函数
def sigmod(inX):
	return 1.0/(1+np.exp(-inX))

# 梯度下降法
def gradAscent(dataMatIn, classLabels):
	dataMat = np.mat(dataMatIn)
	labelMat = np.mat(classLabels).transpose() # 将标签转置为列向量
	m, n = np.shape(dataMat)
	print(dataMat.shape)
	alpha = 0.001 # 步长
	maxCycles = 500 # 迭代次数
	weights = np.ones((n, 1)) # 3x1的矩阵
	for k in range(maxCycles):
		h = sigmod(dataMat * weights)
		error = (labelMat - h)
		# dataMat进行转置, 计算真实类别与预测类别的差值，并按照该差值的方向调整回归系数
		weights += alpha * dataMat.transpose() * error
	return weights

def plotBestFit(dataMat, labelMat, weights):
	dataArr = np.array(dataMat)
	n = np.shape(dataArr)[0]
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
		else:
			xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	# s 表示点的大小size
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
	ax.scatter(xcord2, ycord2, s=30, c='green')
	x = np.arange(-3.0, 3.0, 0.1)
	y = (-weights[0] - weights[1] * x) / weights[2] # 计算x、y的关系式
	ax.plot(x, y)
	plt.xlabel("X1"); plt.ylabel("X2")
	plt.show()

def stocGradAscent0(dataMatrix, classLabels, numIter=150):
	dataArr = np.array(dataMatrix)
	m, n = np.shape(dataMatrix)
	# alpha = 0.01
	weights = np.ones(n)
	for j in range(numIter):
		dataIndex = list(range(m))
		for i in range(m):
			alpha = 4/(1.0+j+i) + 0.01 # 每次都需要调整alpha,不断减小。
			randIndex = int(np.random.uniform(0, len(dataIndex))) # 进行随机更新
			# print randIndex, dataArr[randIndex]
			#print dataArr[i], " * ", weights, " = " , np.sum(dataArr[i] * weights)
			h = sigmod(np.sum(dataArr[randIndex] * weights))
			error = classLabels[randIndex] - h
			weights += alpha * error * dataArr[randIndex]
			del(dataIndex[randIndex])
	return weights

def classifyVector(inX, weights):
	prob = sigmod(np.sum(inX*weights))
	if prob > 0.5:
		return 1.0
	else:
		return 0.0

def colicTest():
	frTrain = open("horseColicTraining.txt")
	frTest = open("horseColicTest.txt")
	trainingSet = []; trainingLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split("\t")
		lineArr = []
		for i in range(21):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[21]))
	trainWeights = stocGradAscent0(np.array(trainingSet), trainingLabels, 500)
	errorCount = 0; numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1.0
		currLine = line.strip().split("\t")
		lineArr = []
		for i in range(21):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
			errorCount += 1
	errorRate = (float(errorCount)/numTestVec)
	print('the error rate of this test is %f' % errorRate)
	return errorRate

def multiTest():
	numTests = 10; errorSum = 0.0
	for _ in range(numTests):
		errorSum += colicTest()
	print('after %d iterations the average error rate is %f' % (numTests, errorSum/float(numTests)))
