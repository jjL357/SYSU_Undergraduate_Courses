import numpy as np
def PCA(data, dim=10):
    # 数据中心化
    X_mean = np.mean(data, axis=0)
    X = data - X_mean
    # 计算协方差矩阵
    cov_matrix = np.cov(X, rowvar=False)
    # SVD 分解
    U, S, Vt = np.linalg.svd(cov_matrix)
    # 选择前 dim 个特征向量
    P = Vt[:dim]
    # 转换数据
    Y = np.dot(X, P.T)
    # print(centered_data.shape,P.shape,Y.shape)
    return Y, P,Vt
