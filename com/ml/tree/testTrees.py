#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/22 12:33
# @Author  : Jasontang
# @Site    : 
# @File    : testTrees.py
# @ToDo    : trees.py 的测试

import trees as trees


def testCreateDataSet():
	myData, labels = trees.createDataSet()
	# print myData
	# print labels

	return myData, labels

def testCalcShannonEnt(dataSet):
	shannonEnt = trees.calcShannonEnt(dataSet)
	print(shannonEnt)
	return shannonEnt

def testChooseBestFeatureToSplit(dataSet):
	# print dataSet
	print(trees.chooseBestFeatureToSplit(dataSet))

if __name__ == '__main__':
	# myData, labels = testCreateDataSet()
	# print myData, labels
	# testCalcShannonEnt(myData)
	# print trees.splitDataSet(myData, 0, 1)
	# print trees.splitDataSet(myData, 0, 0)
	# print trees.splitDataSet(myData, 1, 0)
	# print trees.splitDataSet(myData, 1, 1)
	# testChooseBestFeatureToSplit(myData)
	# print myData
	# myTree = trees.createTree(myData, labels)
	# print myTree
	# treePlotter.createPlot(myTree)
	# myTree = treePlotter.retrieveTree(0)
	# print myTree
	# myTree = treePlotter.retrieveTree(1)
	# print myTree
	# myTree['no surfacing'][3] = 'maybe'
	# print myTree

	# print treePlotter.getNumLeafs(myTree)
	# print treePlotter.getTreeDepth(myTree)
	# treePlotter.createPlot(myTree)
	# # 测试分类结果
	# print labels
	# myTree = treePlotter.retrieveTree(0)
	# print myTree
	#
	# print trees.classify(myTree, labels, [1, 0])
	# print trees.classify(myTree, labels, [1, 1])
	# treePlotter.createPlot(myTree)

	# # 测试存储决策树
	# trees.storeTree(myTree, 'classifierStorage.txt')
	# print trees.grabTree('classifierStorage.txt')
	trees.glasses()