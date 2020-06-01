import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 一维线性回归例子

# 1、create Dataset
x1 = np.linspace(-10, 10, 50)
# 2、定义增广矩阵
X = x1.reshape(-1, 1)
rows, cols = X.shape
X = np.hstack((X, np.ones((rows, 1))))
W = np.array([2,5]).reshape(-1,1)
# fx为理论预测值，Y为样本实际值
Y = np.dot(X, W) + (np.random.randn(50) * 2).reshape(-1,1)
fx = np.dot(X, W)
fig = plt.figure()
#ax3.plot_surface(x1, x2, Y, cmap='rainbow')
plt.scatter(X[:,0], Y, cmap='Blues')

# 3、求解
if np.linalg.det( np.dot(X.transpose(), X) ) == 0:
    print("行列式不可逆")
else:
    print("行列式可逆")
W = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), Y)
print(W)
fx = np.dot(X, W)
plt.plot(X[:,0], fx.reshape(-1), 'gray')

plt.show()

