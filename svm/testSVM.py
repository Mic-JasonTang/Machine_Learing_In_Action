import svm
import numpy as np

dataArr, labelArr = svm.loadDataSet("testSet.txt")
print("dataArr.shape:", np.shape(dataArr), "labelArr.shape:", np.shape(labelArr))

#b, alphas = svm.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)

b, alphas = svm.smoP(dataArr, labelArr, 0.6, 0.001, 40)

print("b=", b)
# print("---")
print("alpha.shape:", alphas.shape)

print(alphas[alphas > 0])

# 支持向量
# for i in range(len(dataArr)):
#     if alphas[i] > 0.0:
#         print(dataArr[i], labelArr[i])

w = svm.calcWs(alphas, dataArr, labelArr)
#svm.plot_sv(dataArr, labelArr, w, b, alphas)

i = 5
y_ = svm.predict(dataArr[5], w, b)
print("y_:{}, y:{}".format(y_, labelArr[i]))

svm.eval(dataArr, labelArr, w, b)

svm.testRbf(1)

svm.testDigits(kTup=('rbf', 5))