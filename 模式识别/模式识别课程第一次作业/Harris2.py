import numpy as np
import cv2

def harris_corner_detection(image, k=0.04, threshold=0.01):
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Sobel算子计算梯度
    dx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算Harris矩阵M的分量
    Ixx = dx ** 2
    Iyy = dy ** 2
    Ixy = dx * dy

    # 使用高斯滤波平滑这些分量
    window_size = 3
    Ixx = cv2.GaussianBlur(Ixx, (window_size, window_size), 0)
    Iyy = cv2.GaussianBlur(Iyy, (window_size, window_size), 0)
    Ixy = cv2.GaussianBlur(Ixy, (window_size, window_size), 0)

    # 计算Harris响应
    det_M = Ixx * Iyy - Ixy ** 2
    trace_M = Ixx + Iyy
    harris_response = det_M - k * (trace_M ** 2)

    # 根据阈值找到角点
    corners = np.zeros_like(gray_image)
    corners[harris_response > threshold * harris_response.max()] = 255

    return corners

# 读取彩色图像
image = cv2.imread('images\\uttower2.jpg')

# 检测Harris角点
corners = harris_corner_detection(image)

# 将角点标记转换回彩色空间以便可视化
corner_markers = cv2.merge((np.zeros_like(corners), np.zeros_like(corners), corners))

# 将角点标记添加到原始图像上
result = cv2.addWeighted(image, 0.5, corner_markers, 0.5, 0)

# 显示结果
cv2.imshow('Harris Corners', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite("results\\uttower2_keypoints.jpg",result)
