import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from PCA import PCA
# 加载图像
image_path = "data\scenery.jpg"
image = Image.open(image_path)
data = np.array(image)

# 降维到不同维度
compression_levels = [10, 50, 100, 150]

for level in compression_levels:
    # 不同通道重建结果
    reconstructed_channels = []
    # 对不同通道进行PCA重建
    for channel in range(data.shape[2]):
        channel_data = data[:,:,channel]
        # 执行 PCA
        compressed_data, components,Vt= PCA(channel_data, level)
        
        # 重建图像
        reconstructed_data = np.dot(compressed_data,  components)
        reconstructed_image_array = (reconstructed_data + np.mean(channel_data, axis=0))
        reconstructed_channels.append(reconstructed_image_array)
     
    # 将三个通道的数据合并为重建的彩色图像
    reconstructed_image = np.stack(reconstructed_channels, axis=-1)   
    reconstructed_image = Image.fromarray(reconstructed_image.astype(np.uint8))
    
    # 可视化原始图像和重建图像
    fig, axes = plt.subplots(1,2, figsize=(18, 6))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(reconstructed_image)
    axes[1].set_title(f'Reconstructed Image (Top {level} components)')
    axes[1].axis('off')
    plt.show()
     # 保存重建图像
    reconstructed_image.save(f"results/PCA/recovered_scenery_top_{level}.jpg")
    fig.savefig(f"results/PCA/recovered_scenery_top_{level}_comparison.jpg")
    plt.close(fig)

