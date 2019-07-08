import adaboost
import numpy as np

# dataMat, labelMat = adaboost.loadSimpData()
# D = np.mat(np.ones((5,1))/5)

# bestStump, minError, bestClasEst = adaboost.buildStump(dataMat, labelMat, D)

# print(bestStump)
# print(minError)
# print(bestClasEst)

# classifierArr = adaboost.adaBoostTrainDS(dataMat, labelMat, 40)

# print(classifierArr)

# result = adaboost.adaClassify([[1, 5], [2, 4]], classifierArr)

# print(result)

dataMat, labelMat = adaboost.loadDataSet("horseColicTraining2.txt")

classifierArr, aggClassEst = adaboost.adaBoostTrainDS(dataMat, labelMat, 50)

print(classifierArr)
print(aggClassEst.T.shape)

adaboost.plotROC(aggClassEst.T, labelMat)

# dataMat, labelMat = adaboost.loadDataSet("horseColicTest2.txt")
# pred = adaboost.adaClassify(dataMat, classifierArr)

# print(np.mat(labelMat).shape)
# print(np.mat(labelMat).T.shape)
# print(len(dataMat))
# errorMat = np.mat(np.ones((len(dataMat), 1)))
# rate = (errorMat[pred != np.mat(labelMat).T].sum() / len(dataMat))

# print(rate)