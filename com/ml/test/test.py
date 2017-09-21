#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/19 22:18
# @Author  : Jasontang
# @Site    : 
# @File    : test.py
# @ToDo    : 测试 kNN.py

import os
import numpy as np
import com.ml.kNN as kNN
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = "SimHei"
mpl.rcParams['axes.unicode_minus'] = False

def testCreateDataSet(show = False):
	group, labels = kNN.createDataSet()
	print "group=\n", group
	print "labels=\n", labels

	if show:
		x = group[:, 0]
		y = group[:, 1]

		plt.figure(figsize=(10,8))
		plt.plot(x, y, "go-.", label="AAAAAA", lw = 2)
		for i in range(len(x)):
			plt.text(x[i] - 0.03, y[i], labels[i])

		plt.title(u"k-近邻算法：带有4个数据点的简单例子")
		plt.legend("upper left")
		plt.show()
	return group, labels

def testClassify0(inX, dataSet, labels):
	return kNN.classify0(inX, dataSet, labels, 3)


def testfile2matrix():
	datingDataMat, datingLabels = kNN.file2matrix(r'../datingTestSet2.txt')
	# print datingDataMat
	# print datingLabels
	# plt.figure(figsize=(6,10))
	# plt.subplot(211)
	# plt.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 50.0* np.array(datingLabels), np.array(datingLabels))
	# plt.xlabel(u'玩视频游戏所耗时间百分比')
	# plt.ylabel(u'每周消费的冰激淋公升数')
	# plt.subplot(212)
	# plt.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 50.0* np.array(datingLabels), np.array(datingLabels))
	# plt.xlabel(u'每年获取的飞行常客里程数')
	# plt.ylabel(u'玩视频游戏所耗时间百分比')
	# plt.show()
	return datingDataMat, datingLabels

if __name__ == '__main__':
	# group, labels = testCreateDataSet(True)
	# inX = list(input("输入数据："))
	# print "属于:", testClassify0(inX, group, labels)
	# print os.getcwd()
	# datingDataMat, datingLabels = testfile2matrix()
	# normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
	# print normMat
	# print ranges
	# print minVals
	# kNN.datingClassTest('../datingTestSet2.txt')
	kNN.classifyPerson()