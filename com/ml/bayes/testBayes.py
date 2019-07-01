#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/12 11:10
# @Author  : Jasontang
# @Site    : 
# @File    : testBayes.py
# @ToDo    : Bayes Driver

import bayes
import numpy as np

if __name__ == '__main__':
	# listOPosts, listClasses = bayes.loadDataSet()
	# myVocabList = bayes.createVocabList(listOPosts)
	# print myVocabList
	#
	# print bayes.setOfWords2Vect(myVocabList, listOPosts[0])
	# print bayes.setOfWords2Vect(myVocabList, listOPosts[3])
	#
	# trainMat = map(lambda postinDoc: bayes.setOfWords2Vect(myVocabList, postinDoc), listOPosts)
	# print
	# for item in trainMat:
	# 	print item
	#
	# p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
	#
	# print p0V
	# print p1V
	# print pAb
	#
	# testEntry = ['love', 'my', 'dalmation']
	# thisDoc = np.array(bayes.setOfWords2Vect(myVocabList, testEntry))
	# # print thisDoc
	# print testEntry, 'classified as:', bayes.classifyNB(thisDoc, p0V, p1V, pAb)
	# testEntry = ['stupid', 'garbage']
	# thisDoc = np.array(bayes.setOfWords2Vect(myVocabList, testEntry))
	# # print thisDoc
	# print testEntry, 'classified as:', bayes.classifyNB(thisDoc, p0V, p1V, pAb)
	result = 0
	N = 1.0
	for i in range(int(N)):
		result += bayes.spamTest()
	print(result/N)
