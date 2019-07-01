#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/18 10:36
# @Author  : Jasontang
# @Site    : 
# @File    : apriori.py
# @ToDo    : 使用apriori算法来发现频繁集

from __future__ import print_function

# 最小支持度阈值:  0.2000--0.6000。 最小可信度阈值:  0.5000--0.9000。
def readfiell(filename):
    res = []
    with open(filename) as f:
        for line in f.readlines():
            items = map(lambda x: int(x), line.strip().split(" "))
            res.append(items)
            print(items)
    return res

def loadDataSet(data_scale="simple"):
    if data_scale == "simple":
        return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    elif data_scale == "mini":
        return readfiell("DATA.txt")
    elif data_scale == "normal":
        return readfiell("T10I4D100K.dat")
    elif data_scale == "large":
        return readfiell("accidents.dat")

# 产生候选集-集合的集合（子集合中都只有一个元素）
# 相当于将数据集中每个元素封装成集合之后放入到大集合中
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])
    # print("before sort C1:", C1)
    C1.sort()
    # print("after sort C1:", C1)
    return map(frozenset, C1)

# 筛选满足支持度要求的集合
def scanD(D, Ck, minSupport):
    ssCnt = {}
    # 支持度计数
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:    # 计算支持度
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)  # 这里任意插入都可以
        supportData[key] = support
    return retList, supportData


# 集合求并集
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    # print("lenLk:", lenLk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            # print(Lk)
            # print(Lk[i])
            # print("list(Lk[i]):", list(Lk[i]))
            #list(Lk[i]) 转换为列表
            # 截取集合开始元素相同的集合
            L1 = list(Lk[i])[:k-2]
            # print("L1---->", L1)
            # print("Lk[j]):", list(Lk[j]))
            L2 = list(Lk[j])[:k-2]
            # print("L2---->", L2)
            # 排序之后，进行合并
            L1.sort(); L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)  # 产生只有1个元素的集合的集合作为候选集
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)  # L1是从C1中挑选满足支持度的集合
    L = [L1]  # 转换为列表, L1为一维列表,元素为集合
    # print("L:", np.shape(L))
    # print("L:", L)
    k = 2
    while len(L[k - 2]) > 0:  # 当Lk为空时退出
        # 生成频繁项集
        Ck = aprioriGen(L[k-2], k)   # 从L中生成具有k个元素集合的集合
        # 挑选满足条件的集合
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k +=1
    return L, supportData


# 生成关联规则
def generateRules(L, supportData, minConf=0.5):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, brl, minConf=0.5):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print("freqSetL", freqSet, "conseq", conseq, ">>>>", freqSet-conseq, "-->", conseq, "conf:", conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if len(freqSet) > (m+1):
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


if __name__ == '__main__':
    dataSet = loadDataSet("simple")
    minSupport = 0.4
    minConf = 0.5
    # print("dataSet:", dataSet)
    #
    # C1 = createC1(dataSet)
    # print("C1:", C1)
    #
    # D = map(set, dataSet)
    # print("D:", D)
    #
    # L1, supportData0 = scanD(D, C1, 0.5)
    # print("L1:", L1)
    # print("supportData0:", supportData0)

    L, supportData = apriori(dataSet, minSupport=minSupport)
    print("L:", L)
    print("supportData:", supportData)
    # print("aprioriGen(): ", aprioriGen(L[0], 2))

    # L, supportData = apriori(dataSet, minSupport=0.7)
    # print("L:", L)

    rules = generateRules(L, supportData, minConf=minConf)

    print("rules:")
    for rule in rules:
        print(rule)
    print("最小支持度:%f, 最小可信度:%f, 总共有%d条记录" % (minSupport, minConf, len(rules)))
