from sklearn import svm

x = [[-1,0], [-1,2], [1,2], [0,0], [1,0], [1,1]]
y = [-1, -1, 1, -1, -1, 1]

clf = svm.SVC(C=10, kernel="linear")
clf.fit(x, y)

print("w:", clf.coef_)
print("b:", clf.intercept_)
print("sv:", clf.support_vectors_)

import matplotlib.pyplot as plt
import numpy as np

x_ = np.arange(-2, 2)
b = clf.intercept_[0]
w = clf.coef_[0]
y_ = (-w[0]*x_ - b) / w[1]

a = -w[0]/w[1]
s = clf.support_vectors_[0]
yy_down = a * x_ + (s[1] - a * s[0])
s = clf.support_vectors_[-1]
yy_up = a * x_ + (s[1] - a * s[0])

plt.scatter([i[0] for i in x], [i[1] for i in x], c=y)
for i in range(len(y)):
    plt.text(x[i][0], x[i][1]+0.1, "({},{})".format(x[i][0], x[i][1]))
plt.plot(x_, y_, 'k-')
plt.plot(x_, yy_up, "k--")
plt.plot(x_, yy_down, "k--")
plt.show()