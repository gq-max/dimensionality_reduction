from load_data import *
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Simhei']  # 使用黑体
mpl.rcParams['axes.unicode_minus'] = False  # 让可以显示负号


def p_pca(data, k):
    p, m = data.shape
    mu = np.mean(data, axis=1).reshape((p, 1))
    data = data - mu
    S = (data @ data.T) / m  # 协方差矩阵
    vector_U, value, vector_V = np.linalg.svd(data)
    sort_indices = np.argsort(-value)
    I = np.eye(k)
    sigma2 = sum(value[sort_indices[k:]]) / (p - k)
    diag_sorted = np.diag(value[sort_indices[:k]])
    W = vector_U[:, 0:k] @ ((diag_sorted - sigma2 * I) ** 0.5)
    Z = np.zeros((k, m))
    for i in range(m):
        Z[:, i:i+1] = np.linalg.inv(W.T @ W + sigma2 * I) @ W.T @ (data[:, i:i+1] - mu)
    recon_data = (W @ Z + mu)
    return Z, recon_data


X = get_data()  # (10304, 400)
Z, recon_X = p_pca(X, 200)
print(Z)
x = recon_X[:, 0].reshape(112, 92)
plt.title(u"ppca重构图片")
plt.imshow(x, cmap='gray')
plt.show()

