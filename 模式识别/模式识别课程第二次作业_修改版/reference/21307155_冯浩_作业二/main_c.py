import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

def my_pca(X, n_components):
    # 数据标准化
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # 避免标准差为零
    X_std = (X - mean) / std

    # 计算协方差矩阵
    cov_matrix = np.cov(X_std, rowvar=False)

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 按特征值大小排序
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # 选择前 n_components 个主成分
    selected_eigenvectors = eigenvectors[:, :n_components]

    # 将数据投影到主成分上
    projected_data = np.dot(X_std, selected_eigenvectors)

    return projected_data

def kmeans(X, k, max_iters, tol):
    def initialize_centers(X, k):
        indices = np.random.choice(X.shape[0], k, replace=False)
        return X[indices]

    def assign_clusters(X, centers):
        distances = np.sqrt(np.sum((X[:, np.newaxis] - centers) ** 2, axis=2))
        return np.argmin(distances, axis=1)

    def update_centers(X, labels, k):
        centers = np.zeros((k, X.shape[1]))
        for i in range(k):
            centers[i] = np.mean(X[labels == i], axis=0)
        return centers

    centers = initialize_centers(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centers)
        new_centers = update_centers(X, labels, k)
        center_diff = np.sum(np.abs(new_centers - centers))
        if center_diff < tol:
            print(f"Converged after {_} iterations.")
            break
        centers = new_centers
    return centers, labels

# 加载 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 将图像展平并进行归一化处理
train_images = train_images.reshape((len(train_images), -1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((len(test_images), -1))
test_images = test_images.astype('float32') / 255

# 合并训练集和测试集数据用于聚类
all_images = np.concatenate((train_images, test_images), axis=0)
all_labels = np.concatenate((train_labels, test_labels), axis=0)

# 使用 PCA 进行降维
n_components = 150  # 选择降维后的维度
all_images_pca = my_pca(all_images, n_components)


# 执行自己实现的 K-means 聚类
centers, labels = kmeans(all_images_pca, 10, 500, 1e-6)

# 显示每个聚类标签的前100个原始图像
plt.figure(figsize=(12, 60))
for i in range(10):
    cluster_images = all_images[labels == i]
    plt.figure(figsize=(12, 6))
    for j in range(min(100, len(cluster_images))):
        plt.subplot(10, 10, j + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(cluster_images[j].reshape(28, 28), cmap='gray')
    plt.suptitle(f'Cluster {i}')
    plt.show()

# 输入新标签并计算准确率
correct_count = 0
incorrect_count = 0

# 用于记录每个聚类的新标签
new_labels = np.zeros_like(labels)

for i in range(10):
    new_label = int(input(f"Enter a new label for Cluster {i} (0-9): "))
    new_labels[labels == i] = new_label

# 计算准确率
correct_count = np.sum(new_labels == all_labels)
incorrect_count = len(all_labels) - correct_count

# 输出结果
print(f'Correct count: {incorrect_count}')
print(f'Incorrect count: {correct_count}')
accuracy = correct_count / len(all_labels) if len(all_labels) > 0 else 0
print(f'Accuracy: {1-accuracy:.2%}')
