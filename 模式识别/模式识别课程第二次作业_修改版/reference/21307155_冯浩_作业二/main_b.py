import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

# 计算峰值信噪比（PSNR）
def calculate_psnr(original, reconstructed, channelwise=False):
    if channelwise:
        psnrs = []
        for channel in range(original.shape[-1]):
            mse_channel = np.mean((original[:, :, channel] - reconstructed[:, :, channel]) ** 2)
            max_pixel = 255  # 对于8位图像，最大像素值为255
            psnr_channel = 10 * np.log10((max_pixel ** 2) / mse_channel)
            psnrs.append(psnr_channel)
        return np.mean(psnrs)
    else:
        mse = np.mean((original - reconstructed) ** 2)
        max_pixel = 255  # 对于8位图像，最大像素值为255
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return psnr

# 加载 lena.png 彩色图像
lena_image = Image.open('作业二/data/scenery.jpg')
lena_data = np.array(lena_image)

# 显示原始图像
plt.imshow(lena_data.astype('uint8'))
plt.axis('off')
plt.title('Original Lena Image')
plt.show()

# 压缩和重建 Lena 图像
compress_dimensions = [10, 50, 100, 150]
for dim in compress_dimensions:
    # 提取 RGB 通道数据并分别进行 PCA 压缩和重建
    reconstructed_channels = []
    for channel in range(3):
        channel_data = lena_data[:, :, channel]
        projected_data, components = my_pca(channel_data, dim)
        reconstructed_channel = np.dot(projected_data, components)
        reconstructed_channel = reconstructed_channel + np.mean(channel_data, axis=0)
        reconstructed_channels.append(reconstructed_channel)

    # 将三个通道的数据合并为重建的彩色图像
    reconstructed_image = np.stack(reconstructed_channels, axis=-1)

    # 计算整体图像的PSNR（按通道分别计算）
    psnr = calculate_psnr(lena_data, reconstructed_image, channelwise=True)
    print(f'Average PSNR for {dim} components: {psnr:.2f} dB')

    # 将重建后的 Lena 图像保存为文件
    reconstructed_image = Image.fromarray(reconstructed_image.astype('uint8'))
    #reconstructed_image.save(f'results/recovered_lena_top_{dim}.jpg')

    # 显示重建图像
    plt.imshow(reconstructed_image)
    plt.axis('off')
    plt.title(f'Reconstructed Lena Image (Top {dim} Components)\nAverage PSNR: {psnr:.2f} dB')
    plt.show()
