#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/12 10:56
# @Author  : Jasontang
# @Site    : 
# @File    : bayes.py
# @ToDo    : 使用朴素贝叶斯来进行文本分类
import random

import numpy as np

# 创建数据集
def loadDataSet():
	postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
				 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
	return postingList,classVec

# 去掉重复词语,构造词汇表
def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet |= set(document) # 两个集合求并集
	return list(vocabSet)

# 词集模型
def setOfWords2Vect(vocabList, inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1 # 表示VocabList中的单词在inputSet中是否出现
		else:
			print('the word: %s is not in my Vocabulary!' % word)
	return returnVec

# 词袋模型
def bagOfWwords2VecMN(vocabList, inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1  # 累计次数
	return returnVec

# trainMatrix：二维矩阵，每一行是这句话这词汇表中是否出现（01）
# trainCategory: 每篇文档类别标签所构成的向量 listClasses
def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = np.sum(trainCategory) / float(numTrainDocs) # 计算类别1的概率P(1)

	# p0Num = np.zeros(numWords); p1Num = np.zeros(numWords)  # 分子
	# p0Denom = 0.0; p1Denom = 0.0  # 分母

	# 为了避免其中一个概率值为0导致最后结果也是0，则将所有词的出现数初始化为1，分母初始化为2
	p0Num = np.ones(numWords); p1Num = np.ones(numWords)
	p0Denom = 2.0; p1Denom = 2.0

	for i in range(numTrainDocs):
		if trainCategory[i] == 1: # 在给定文档类别条件下词汇表中单词的出现概率。
			p1Num += trainMatrix[i]
			# print 'trainMatrix[i]=\n', trainMatrix[i]
			# print 'p1Num=\n', p1Num
			p1Denom += sum(trainMatrix[i]) # 这一步在 统计出现的总次数
			# print p1Denom
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])

	# p1Vect = p1Num / p1Denom     # 对每一个元素做除法，计算的是p(wi|1)
	# p0Vect = p0Num / p0Denom     # 对每一个元素做除法，计算的是p(wi|0)

	#为了避免下溢出，即很多个很小的小数相乘，最后结果是0，在这里进行取对数操作
	p1Vect = np.log(p1Num/p1Denom)
	p0Vect = np.log(p0Num/p0Denom)

	return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):
	# print np.sum(vec2Classify * p1Vect)
	p1 = np.sum(vec2Classify * p1Vect) + np.log(pClass1)
	p0 = np.sum(vec2Classify * p0Vect) + np.log(1 - pClass1)
	# print np.sum(vec2Classify * p0Vect)
	if p1 > p0:
		return 1
	else:
		return 0

# 解析文本
# 分隔文本，并去掉长度小于2的文本
def textParse(bigString):
	import re
	pattern = re.compile("\W*")
	listOfTokens = re.split(pattern, bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
	docList = []
	classList = []
	fullText = []
	for i in range(1, 26):
		wordList = textParse(open('email/spam/%d.txt' % i, encoding="ISO-8859-1").read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(open('email/ham/%d.txt' % i, encoding="ISO-8859-1").read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	print("vocabList: \n",vocabList)
	print(len(vocabList))
	# 有50个样本
	trainingSet = list(range(50)); testSet = []
	# 随机生成10个样本索引
	for i in range(10):
		randIndex = int(random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat = []; trainClasses = []
	# 构造训练集
	for docIndex in trainingSet:
		indexCounter = bagOfWwords2VecMN(vocabList, docList[docIndex])
		print(indexCounter)
		trainMat.append(indexCounter)
		trainClasses.append(classList[docIndex])
	# 进行训练, 得到分类为0中词出现的概率p0，得到分类为1中词出现的概率p1，以及垃圾邮件的概率
	p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))

	# 进行测试
	errorCount = 0
	for docIndex in testSet:
		wordVector = bagOfWwords2VecMN(vocabList, docList[docIndex])
		if classifyNB(wordVector, p0V, p1V, pSpam) != classList[docIndex]:
			errorCount +=1
			print(docIndex)
	p_result = float(errorCount)/len(testSet)
	print('the error rate is:', p_result)
	return p_result