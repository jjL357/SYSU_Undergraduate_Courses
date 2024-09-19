import numpy as np
import cv2

def harris_corner_detection(image, k=0.04, threshold=0.01):
    # 用Sobel算子计算图像的梯度
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算Harris矩阵M的三个分量
    Ixx = dx ** 2
    Iyy = dy ** 2
    Ixy = dx * dy

    # 使用高斯滤波平滑上述三个分量
    window_size = 3
    Ixx = cv2.GaussianBlur(Ixx, (window_size, window_size), 0)
    Iyy = cv2.GaussianBlur(Iyy, (window_size, window_size), 0)
    Ixy = cv2.GaussianBlur(Ixy, (window_size, window_size), 0)

    # 计算Harris响应
    det_M = Ixx * Iyy - Ixy ** 2 # 行列式 = 特征值的乘积
    trace_M = Ixx + Iyy # 迹 = 特征值的和
    harris_response = det_M - k * (trace_M ** 2)

    # 根据阈值找到角点并进行标记
    corners = np.zeros_like(image)
    corners[harris_response > threshold * harris_response.max()] = 255

    return corners

# 读取图像
image = cv2.imread('images\\uttower1.jpg', cv2.IMREAD_GRAYSCALE)

# 进行Harris角点检测
corners = harris_corner_detection(image)

# 将角点标记为红色
result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
result[corners == 255] = [0, 0, 255]

# 将原图和结果图放在一起对比显示
comparison = np.hstack((cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), result))
# cv2.imwrite("imagetmp/comparison1.png",comparison)

# 显示结果
cv2.imshow('Harris Corners', comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()


# cv2.imwrite("results/sudoku_keypoints.png",result)
