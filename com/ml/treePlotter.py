#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/23 21:23
# @Author  : Jasontang
# @Site    : 
# @File    : treePlotter.py
# @ToDo    : 使用Matplotlib注解绘制图形

import matplotlib as mpl
import matplotlib.pyplot as plt

# 定义树结点格式的常量
decisionNode = dict(boxstyle='sawtooth', fc="0.8")
leafNode = dict(boxstyle="round4", fc='0.8')
arrow_args = dict(arrowstyle="<-")

mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = "SimHei"

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	# 定义一个绘图区，该区域由全局变量createPlot.ax1定义(Python语言中所有的变量默认都是全局有效的)
	createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords="axes fraction",
							xytext=centerPt, textcoords="axes fraction",
							va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
# 主函数
def createPlot(inTree):
	# createPlot.ax1 是个变量
	axprops = dict(xticks=[], yticks=[])
	createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
	# 全局变量存储树的宽度和高度
	plotTree.totalW = float(getNumLeafs(inTree))
	plotTree.totalD = float(getTreeDepth(inTree))
	# 用来追踪已经绘制的结点位置，以及防止下一个结点的恰当位置
	plotTree.xOff = -0.5/ plotTree.totalW
	plotTree.yOff = 1.0
	plotTree(inTree, (0.5, 1.0), '')
	plt.show()
	# plotNode(u"决策结点", (0.5, 0.1), (0.1, 0.5), decisionNode)
	# plotNode(u"叶结点", (0.8, 0.1), (0.3, 0.8), leafNode)
	# plt.show()

# 获取叶子节点数量
def getNumLeafs(myTree):
	numLeafs = 0
	firstStr = myTree.keys()[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			numLeafs += getNumLeafs(secondDict[key])
		else:
			numLeafs += 1
	return numLeafs

# 获取树的层数
def getTreeDepth(myTree):
	maxDepth = 0
	firstStr = myTree.keys()[0]
	secondDict = myTree[firstStr]
	# print 'secondDict:', secondDict
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			thisDepth = 1 + getTreeDepth(secondDict[key])
		else:
			thisDepth = 1
		if thisDepth > maxDepth: maxDepth = thisDepth
	return maxDepth

# 存储了书的结构
def retrieveTree(i):
	listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
				   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
	return listOfTrees[i]

# 画两点之间线上的注释
def plotMidText(cntrPt, parentPt, txtString):
	# 在父子节点间填充文本信息
	xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
	yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
	createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
	# 计算宽与高
	numLeafs = getNumLeafs(myTree)
	depth = getTreeDepth(myTree)
	firstStr = myTree.keys()[0]
	cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
	# 标记子节点属性值
	plotMidText(cntrPt, parentPt, nodeTxt)
	plotNode(firstStr, cntrPt, parentPt, decisionNode)
	secondDict = myTree[firstStr]
	# 减少y偏移
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD

	for key in secondDict.keys():
		# 如果不是叶子结点，就递归
		if type(secondDict[key]).__name__ == "dict":
			plotTree(secondDict[key], cntrPt, str(key))
		else: # 如果是叶节点,就画出图形
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

