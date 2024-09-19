import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from PIL import Image
from PCA import PCA
# 读取数据
data = scipy.io.loadmat("data/faces.mat")
faces = data['X']

# 展示前 49 个主成分
fig, axes = plt.subplots(7, 7, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    data = faces[i].reshape(32,32).T
    _,_,Vt = PCA(data) # PCA获取主成分
    ax.imshow(Vt, cmap='gray')
    ax.axis('off')
plt.suptitle('First 49 Principal Components', fontsize=16)
plt.show()
plt.close()
fig.savefig("results/PCA/eigen_faces.jpg")

# 展示前 100 张原图
fig, axes = plt.subplots(10, 10, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    data = faces[i].reshape(32,32).T 
    ax.imshow(data, cmap='gray')
    ax.axis('off')
plt.suptitle('First 100 Original faces', fontsize=16)
plt.show()
plt.close()
fig.savefig("results/PCA/first_100_original_faces.jpg")    

# 降维维度
compression_levels = [10, 50, 100, 150]

# 对原图进行PCA降维重建
for level in compression_levels:
    # 执行 PCA
    compressed_data, components,Vt = PCA(faces,level)
    # 重建图像
    reconstructed_data = np.dot(compressed_data,  components)
    reconstructed_image_array = (reconstructed_data + np.mean(faces, axis=0))
    #reconstructed_image = Image.fromarray(reconstructed_image_array.astype(np.uint8))
    
    # 展示前 100 张原图PCA降维重建结果
    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        data = reconstructed_image_array[i].reshape(32,32).T 
        ax.imshow(data, cmap='gray')
        ax.axis('off')
    plt.suptitle(f'recovered_faces_top_{level}', fontsize=16)
    plt.show()
    plt.close()
    fig.savefig(f"results/PCA/recovered_faces_top_{level}.jpg")  