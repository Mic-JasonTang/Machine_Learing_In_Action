#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/19 22:18
# @Author  : Jasontang
# @Site    : 
# @File    : test.py
# @ToDo    : 测试 kNN.py

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

def testClassify0(dataSet, labels):
	return kNN.classify0([0, 0], dataSet, labels, 3)


if __name__ == '__main__':
	group, labels = testCreateDataSet()
	print testClassify0(group, labels)

