#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/22 12:27
# @Author  : Jasontang
# @Site    : 
# @File    : trees.py
# @ToDo    : 决策树

from math import log
import treePlotter as treePlotter
import operator

# 计算香农熵
def calcShannonEnt(dataSet):
	numEntris = len(dataSet)
	labelCounts = {}
	for featVect in dataSet:
		currentLabel = featVect[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1

	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntris
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt

# 创建临时数据集
def createDataSet():
	dataSet = [[1, 1, 'yes'],
			   [1, 1, 'yes'],
			   [1, 0, 'no'],
			   [0, 1, 'no'],
			   [0, 1, 'no']]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels

# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			# 在axis维特征下，划分数据
			reducedFeatVec = featVec[:axis]
			# print "axis=%d" % axis, reducedFeatVec
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet


def chooseBestFeatureToSplit(dataSet):
	# 取特征数量，最后一个是类别
	numFeatures = len(dataSet[0]) - 1
	# 计算所有数据的熵
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0; bestFeature = -1
	for i in range(numFeatures):
		# 获取特征
		featList = [example[i] for example in dataSet]
		# print featList
		uniqueVals = set(featList)
		# print uniqueVals
		newEntropy = 0.0
		for value in uniqueVals:
			# 计算条件熵
			# 划分子集
			subDataSet = splitDataSet(dataSet, i, value)
			# print subDataSet
			# 计算子集的概率
			prob = len(subDataSet) / float(len(dataSet))
			# 计算条件熵
			newEntropy += prob * calcShannonEnt(subDataSet)
		# 计算信息增益
		infoGain = baseEntropy - newEntropy
		if infoGain > bestInfoGain:
			bestInfoGain = infoGain
			bestFeature = i

	return bestFeature

# 返回出现次数最多的分类名称
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys(): classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

# 创建树的函数（ID3算法）
def createTree(dataSet, labels):
	# 获取所有的分类
	classList = [example[-1] for example in dataSet]
	# print 'classList=',classList
	# 统计列表中标签是否完全相同
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	# 遍历完了所有特征，仍不能将数据集划分成仅包含唯一类别的分组
	# dataSet[0]=1，表示特征集为空，此时T为单节点树，返回数据集中实例数最大的类作为该结点的类标记
	if len(dataSet[0]) == 1:
		# 则挑选出出现最多的类别
		return majorityCnt(classList)

	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}}
	# print "del=", labels[bestFeat]
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		# print "bestFeat=" , bestFeat, "value=", value
		# print splitDataSet(dataSet, bestFeat, value)
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
	return myTree


# 分类
def classify(inputTree, featLabels, testVec):
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	# print 'secondDict.keys:', secondDict.keys()
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			# print "secondDict[key]:", secondDict[key]
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key], featLabels, testVec)
			else :
				classLabel = secondDict[key]
	return classLabel

# 存储决策树
def storeTree(inputTree, filename):
	import pickle
	fw = open(filename, 'w')
	pickle.dump(inputTree, fw)
	fw.close()

# 恢复决策树
def grabTree(filename):
	import pickle
	fr = open(filename)
	return pickle.load(fr)

def glasses():
	fr = open('../lenses.txt')
	lenses = [inst.strip().split('\t') for inst in fr.readlines()]
	lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
	lensesTree = createTree(lenses, lensesLabels)
	print lensesTree
	treePlotter.createPlot(lensesTree)
