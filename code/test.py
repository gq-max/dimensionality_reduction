import numpy as np
import scipy.io as scio
from matplotlib import image
import matplotlib.pyplot as plt
from load_data import *
# # 特征分解
# a = np.array([[-1, 1, 0],
#               [-4, 3, 0],
#               [1, 0, 2]])
# eigenvalue, eigenvector = np.linalg.eig(a)
# print(eigenvalue)
# print(eigenvector)

# # 数据中心化处理
# a = np.array([[1, 2, 3],
#               [2, 2, 2],
#               [3, 1, 1],
#               [3, 4, 5]])
# a_mean = np.mean(a, axis=1)
# a_mean = a_mean.reshape(4, 1)
# print(a_mean)
# a = a - a_mean
# print(a)


# a = np.array([[1.2, 3.2, 2.5],
#               [5.2222, 3.1111, 4.22134]])
# a = a.astype(np.int64)
# print(a)

# def matRecover(temp1, temp2):
#     data = np.zeros((len(temp1), len(temp2)))
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             data[i, j] = temp1[i] * temp2[j]
#     return data
#
# def pca_svd(im_temp, k):
#     u_temp, s_temp, v_temp = np.linalg.svd(im_temp)
#     im_res = np.zeros(im_temp.shape)
#     v_index = np.argsort(-s_temp)
#     for i in range(k):
#         temp1 = u_temp[:, v_index[i]].tolist()
#         temp2 = v_temp[v_index[i]].tolist()
#         temp3 = matRecover(temp1, temp2)
#         im_res = im_res + s_temp[v_index[i]] * temp3
#     return im_res
#
# X = get_data()
# X = X.astype(np.float32)
# p, m = X.shape
# x_mean = np.mean(X, axis=1).reshape((p, 1))
# X = X - x_mean
# res = pca_svd(X, 150)
# x = res[:, 0].astype(np.int32).reshape(112, 92)
# plt.imshow(x, cmap='gray')
# plt.show()
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from load_data import *
#
# X = get_data().T
# features = StandardScaler().fit_transform(X)
# pca = PCA(n_components=0.95, whiten=True)  # 保留百分之**的信息
# features_pca = pca.fit_transform(features)
# print(features.shape)
# print(features_pca.shape)

# x = np.array([1, 3, 4, 2, 5])
# index = [4, 2, 1, 3, 0]
# print(sum(x[index[2:]]))
# print(np.diag(x))
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
    diag_sorted = np.diag(value[sort_indices[:k]])
    diag_sorted2 = np.diag(value[sort_indices[:k]]**(-0.5))
    W = vector_U[:, 0:k] @ (diag_sorted ** 0.5)
    print(diag_sorted.shape)
    Z = (diag_sorted2) @ vector_U[:, 0:k].T @ data
    recon_data = (W @ Z + mu)
    return Z, recon_data


X = get_data()  # (10304, 400)
Z, recon_X = p_pca(X, 200)
print(Z)
x = recon_X[:, 0].reshape(112, 92)
plt.title(u"ppca重构图片")
plt.imshow(x, cmap='gray')
plt.show()


