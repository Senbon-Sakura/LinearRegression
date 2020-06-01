import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 二维线性回归例子
# 1、create Dataset, 初始化时不要可以取整数，否则会导致矩阵求逆时出现不可逆或者逆矩阵数值太大，计算结果误差很大
x1 = np.linspace(1, 4, 50)
x2 = np.linspace(-8, 4, 50)
# 2、定义增广矩阵
X = np.hstack((x1.reshape(-1,1), x2.reshape(-1,1))) # shape=(2,50)
rows, cols = X.shape
X = np.hstack((X, np.ones((rows, 1))))
W = np.array([1,2,5]).reshape(-1,1)
# fx为理论预测值，Y为样本实际值
Y = np.dot(X, W) + (np.random.randn(rows) * 0.5).reshape(-1,1)
fx = np.dot(X, W)
fig = plt.figure()
ax3 = plt.axes(projection='3d')
#ax3.plot_surface(x1, x2, Y, cmap='rainbow')
ax3.scatter3D(x1, x2, Y, cmap='Blues')
ax3.plot3D(X[:,0], X[:,1], fx.reshape(-1), 'gray')

# 3、求解
if np.linalg.det( np.dot(X.transpose(), X) ) == 0:
    print("行列式不可逆")
else:
    print("行列式可逆")
XTX = np.dot(X.transpose(), X)
XTXI = np.linalg.inv(XTX)
XTXIXT = np.dot(XTXI, X.transpose())
W = np.dot(XTXIXT, Y)
print(W)
fx = np.dot(X, W)
#ax3.plot3D(X[:,0], X[:,1], fx.reshape(-1), 'gray')

plt.show()


