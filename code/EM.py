from load_data import *
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Simhei']  # 使用黑体
mpl.rcParams['axes.unicode_minus'] = False  # 让可以显示负号


def EM_pca(data, k):
    p, m = data.shape
    # 初始化
    W = np.random.randn(p, k)
    Z = np.random.randn(k, m)
    x_mean = np.mean(data, axis=1).reshape(p, 1)
    for epoch in range(50):
        print(epoch)
        # E步
        x_mean = np.mean(data, axis=1).reshape(p, 1)
        data = data - x_mean
        Z = np.linalg.inv(W.T @ W) @ W.T @ data
        # M步
        W = data @ Z.T @ np.linalg.inv(Z @ Z.T)
    recon_data = (W @ Z + x_mean)
    return Z, recon_data


X = get_data()  # (10304, 400)
Z, recon_X = EM_pca(X, 200)
print(Z)
x = recon_X[:, 0].reshape(112, 92)
plt.title(u"EM_pca重构图片")
plt.imshow(x, cmap='gray')
plt.show()
