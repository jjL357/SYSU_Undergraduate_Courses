import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from PIL import Image

# 定义 PCA 函数
def my_pca(X, n_components):
    # 数据标准化
    X_std = X - np.mean(X, axis=0)

    # 计算协方差矩阵
    cov_matrix = np.cov(X_std, rowvar=False)

    # SVD 分解
    _, _, Vt = np.linalg.svd(cov_matrix)

    # 选择主成分
    components = Vt[:n_components]

    # 数据投影
    projected_data = np.dot(X_std, components.T)

    return projected_data, components

# 加载 Eigen Face 数据集
data = scipy.io.loadmat('data/faces.mat')
faces_data = data['X']
# print(faces_data)
for i in range(5):
        plt.imshow(faces_data[i].reshape(32, 32).T, cmap='gray')
        plt.axis('off')
        # plt.title(f'Original Face {i+1}')
        plt.savefig(f'results/original_face_{i+1}.jpg')
        plt.close()
# 创建一个空白的大图用于合成原始图像
big_original_image = Image.new('RGB', (320, 320))

# 设置每个小图像的位置和尺寸，并添加到大图上
for i in range(100):
    col = i % 10
    row = i // 10
    extent = (col * 32, row * 32, (col + 1) * 32, (row + 1) * 32)
    img_data = faces_data[i].reshape((32, 32)).T  # 转置矩阵以正确显示图像
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))  # 将数据归一化到0-1范围
    img_data = (img_data * 255).astype(np.uint8)  # 转换为8位无符号整数
    original_image = Image.fromarray(img_data)
    big_original_image.paste(original_image, extent)

# 保存合成的大图
big_original_image.save('results/combined_original_faces.jpg')
# 压缩和重建人脸数据
compress_dimensions = [10, 50, 100, 150]
for dim in compress_dimensions:
    projected_data, components = my_pca(faces_data, dim)

    # 重建数据
    reconstructed_data = np.dot(projected_data, components)
    reconstructed_data = reconstructed_data + np.mean(faces_data, axis=0)
    psnrs = []
    # 将重建后的人脸保存为图片
    for i in range(len(faces_data)):
        # 计算峰值信噪比 PSNR
        mse = np.mean((faces_data[i] - reconstructed_data[i]) ** 2)
        max_pixel = 255
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        psnrs.append(psnr)
        if i<5:
            # print(mse, max_pixel)
            plt.imshow(reconstructed_data[i].reshape(32, 32).T, cmap='gray')
            plt.axis('off')
            plt.title(f'Compressed and Reconstructed Face {i+1} (Top {dim})\nPSNR: {psnr:.2f} dB')
            plt.savefig(f'results/recovered_faces_top_{dim}_face_{i+1}.jpg')
            plt.close()
    print('dimension is', dim, ':', np.mean(psnrs))
    big_reconstructed_image = Image.new('RGB', (320, 320))

    # 设置每个小图像的位置和尺寸，并添加到大图上
    for i in range(100):
        col = i % 10
        row = i // 10
        extent = (col * 32, row * 32, (col + 1) * 32, (row + 1) * 32)
        img_data = reconstructed_data[i].reshape((32, 32)).T  # 转置矩阵以正确显示图像
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))  # 将数据归一化到0-1范围
        img_data = (img_data * 255).astype(np.uint8)  # 转换为8位无符号整数
        reconstructed_image = Image.fromarray(img_data)
        big_reconstructed_image.paste(reconstructed_image, extent)

    # 保存合成的大图
    big_reconstructed_image.save(f'results/combined_reconstructed_faces_{dim}.jpg')
