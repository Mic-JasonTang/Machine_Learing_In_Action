#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/26 0:01
# @Author  : Jasontang
# @Site    : 
# @File    : test.py
# @ToDo    : 测试LogRegres

import logRegres as logRegres

if __name__ == '__main__':
	# dataMat, labelMat = logRegres.loadDataSet()
	# weights = logRegres.gradAscent(dataMat, labelMat)
	# weights = logRegres.stocGradAscent0(dataMat, labelMat, 20)
	# print weights
	# logRegres.plotBestFit(dataMat, labelMat, weights)
	logRegres.multiTest()