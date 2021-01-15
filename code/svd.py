from load_data import *
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Simhei']  # 使用黑体
mpl.rcParams['axes.unicode_minus'] = False  # 让可以显示负号

X = get_data()  # (10304, 400)
X = X.astype(np.float32)
p, m = X.shape
x_mean = np.mean(X, axis=1).reshape((p, 1))
X = X - x_mean
A = np.dot(X.T, X) / m  # (400, 400)
lamda, V = np.linalg.eig(A)  # A的特征值以列的形式显示
for i in range(m):
    V[:, i:i+1] /= np.dot(V[:, i:i+1].T, V[:, i:i+1])
sorted_indices = np.argsort(-lamda)
chance = [8, 20, 50, 100, 150, 200, 250, 300]  # 降维列表
data = []  # 用来保留降维后重构的数据
for k in chance:
    print("降到{}维，信息量保留为{}".format
          (k, np.sum([lamda[i] for i in range(k)] / np.sum(list(lamda)))))
    U = np.ones((10304, k))
    for i, j in zip(sorted_indices[0:k], range(k)):
        U[:, j:j + 1] = (X @ V[:, i:i + 1]) / np.sqrt(lamda[i])
    Z = U.T @ X
    np.savetxt("result/{}维W值.txt".format(k), U[:, 0:k])
    np.savetxt("result/{}维Z值.txt".format(k), Z)
    X1 = U @ Z + x_mean
    X1 = X1.astype(np.int64)
    data.append(X1[:, 0])
data = np.array(data)
x = data[5].reshape(112, 92)
plt.title(u"ppca重构")
plt.imshow(x, cmap='gray')
plt.show()
# plt.subplot(241)
# x = data[0].reshape(112, 92)
# plt.title(u"8维")
# plt.imshow(x, cmap='gray')
# plt.subplot(242)
# x = data[1].reshape(112, 92)
# plt.title(u"20维")
# plt.imshow(x, cmap='gray')
# plt.subplot(243)
# x = data[2].reshape(112, 92)
# plt.title(u"50维")
# plt.imshow(x, cmap='gray')
# plt.subplot(244)
# x = data[3].reshape(112, 92)
# plt.title(u"100维")
# plt.imshow(x, cmap='gray')
# plt.subplot(245)
# x = data[4].reshape(112, 92)
# plt.title(u"150维")
# plt.imshow(x, cmap='gray')
# plt.subplot(246)
# x = data[5].reshape(112, 92)
# plt.title(u"200维")
# plt.imshow(x, cmap='gray')
# plt.subplot(247)
# x = data[6].reshape(112, 92)
# plt.title(u"250维")
# plt.imshow(x, cmap='gray')
# plt.subplot(248)
# x = data[7].reshape(112, 92)
# plt.title(u"300维")
# plt.imshow(x, cmap='gray')
# plt.show()
