import cv2
import numpy as np

def harris_corner_detection(image, threshold=0.01, window_size=3, k=0.04):
    # 1. 计算图像的梯度
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # 2. 计算 Harris 角点响应函数
    Ixx = dx ** 2
    Ixy = dx * dy
    Iyy = dy ** 2

    height, width = image.shape
    offset = window_size // 2
    corner_response = np.zeros((height, width))

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            window_Ixx = Ixx[y - offset : y + offset + 1, x - offset : x + offset + 1]
            window_Ixy = Ixy[y - offset : y + offset + 1, x - offset : x + offset + 1]
            window_Iyy = Iyy[y - offset : y + offset + 1, x - offset : x + offset + 1]

            # 计算局部窗口内的梯度协方差矩阵的特征值
            Sxx = np.sum(window_Ixx)
            Sxy = np.sum(window_Ixy)
            Syy = np.sum(window_Iyy)

            # 计算角点响应函数值
            det = Sxx * Syy - Sxy ** 2
            trace = Sxx + Syy
            corner_response[y, x] = det - k * trace ** 2

    # 3. 对角点响应函数进行阈值处理
    corner_response[corner_response < threshold * np.max(corner_response)] = 0

    # 4. 非极大值抑制
    corner_response = cv2.dilate(corner_response, None)

    return corner_response

# 读取图像
imagePath = 'Homework 1\\images\\uttower2.jpg'
image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

# 执行 Harris 角点检测
corner_response = harris_corner_detection(image)

# 将检测到的角点标记在原始图像上
image_with_keypoints = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
image_with_keypoints[corner_response > 0.01 * corner_response.max()] = [0, 0, 255]

# 保存结果图像
savePath = 'Homework 1\\results\\uttower2.jpg'
cv2.imwrite(savePath, image_with_keypoints)
