import numpy as np
import torch


def read_pgm(filename):
    '''
    读取pgm文件内容，并将其转换为行向量
    :param filename: 文件名
    :return: 10304个特征的样本值
    '''
    f = open(filename, 'rb')
    f.readline()  # P5\n
    (width, height) = [int(i) for i in f.readline().split()]
    depth = int(f.readline())
    data = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(ord(f.read(1)))
        data.append(row)
    data = np.array(data)
    data = data.reshape(width * height)
    return data


def get_data():
    """得到数据集X，其中X的每列为一个样本， 每一行为一个特征，X为p*n的矩阵,并且数据进行了居中处理"""
    X = []
    for i in range(1, 41):
        for j in range(1, 11):
            fn = "orl_faces/s{}/{}.pgm".format(i, j)
            data = read_pgm(fn)
            X.append(data)
    X = np.array(X)
    X = X.T
    return X


def tensor_data():
    """返回值为在GPU上运行的张量，size为[40, 10, 112, 92]"""
    X_cuda = torch.Tensor(40, 10, 10304)
    for i in range(40):
        for j in range(10):
            fn = "orl_faces/s{}/{}.pgm".format(i + 1, j + 1)
            data = read_pgm(fn)
            X_cuda[i, j] = torch.from_numpy(data)
    X_cuda = X_cuda.cuda()
    return X_cuda
