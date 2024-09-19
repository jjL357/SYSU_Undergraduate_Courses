import cv2
import numpy as np

# 读取图像
L = cv2.imread('images\\uttower1.jpg')  # queryImage
R = cv2.imread('images\\uttower2.jpg')  # trainImage

# 高斯滤波
L = cv2.GaussianBlur(L, (3, 3), 0)
R = cv2.GaussianBlur(R, (3, 3), 0)

# 灰度化
L_gray = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
R_gray = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)

# Sobel 边缘检测
L_dx = cv2.Sobel(L_gray, cv2.CV_64F, 1, 0, ksize=3)
L_dy = cv2.Sobel(L_gray, cv2.CV_64F, 0, 1, ksize=3)
R_dx = cv2.Sobel(R_gray, cv2.CV_64F, 1, 0, ksize=3)
R_dy = cv2.Sobel(R_gray, cv2.CV_64F, 0, 1, ksize=3)

# 计算梯度幅值和方向
L_magnitude, L_angle = cv2.cartToPolar(L_dx, L_dy)
R_magnitude, R_angle = cv2.cartToPolar(R_dx, R_dy)

# 定义 HOG 单元格大小和方向直方图数量
cell_size = 8
bins = 9

# 计算图像大小和单元格数量
L_height, L_width = L_gray.shape
R_height, R_width = R_gray.shape
L_cells_x = L_width // cell_size
L_cells_y = L_height // cell_size
R_cells_x = R_width // cell_size
R_cells_y = R_height // cell_size

# 初始化 HOG 描述符
L_hog_descriptor = np.zeros((L_cells_y, L_cells_x, bins))
R_hog_descriptor = np.zeros((R_cells_y, R_cells_x, bins))

# 计算 HOG 描述符
for i in range(L_cells_y):
    for j in range(L_cells_x):
        cell_magnitude = L_magnitude[i * cell_size:(i + 1) * cell_size,
                                     j * cell_size:(j + 1) * cell_size]
        cell_angle = L_angle[i * cell_size:(i + 1) * cell_size,
                             j * cell_size:(j + 1) * cell_size]
        cell_histo = np.zeros((bins,))
        for m in range(cell_size):
            for n in range(cell_size):
                angle = cell_angle[m, n]
                magnitude = cell_magnitude[m, n]
                bin_idx = int(angle / (360 / bins))
                cell_histo[bin_idx] += magnitude
        L_hog_descriptor[i, j, :] = cell_histo

for i in range(R_cells_y):
    for j in range(R_cells_x):
        cell_magnitude = R_magnitude[i * cell_size:(i + 1) * cell_size,
                                     j * cell_size:(j + 1) * cell_size]
        cell_angle = R_angle[i * cell_size:(i + 1) * cell_size,
                             j * cell_size:(j + 1) * cell_size]
        cell_histo = np.zeros((bins,))
        for m in range(cell_size):
            for n in range(cell_size):
                angle = cell_angle[m, n]
                magnitude = cell_magnitude[m, n]
                bin_idx = int(angle / (360 / bins))
                cell_histo[bin_idx] += magnitude
        R_hog_descriptor[i, j, :] = cell_histo

# 使用 BFMatcher 进行匹配
bf = cv2.BFMatcher()
matches = bf.match(L_hog_descriptor.ravel(), R_hog_descriptor.ravel())

# 进行匹配筛选
BetterChoose1 = []
for m in matches:
    # 进行自定义的匹配筛选条件，这里可以根据实际情况调整
    if m.distance < 50:
        BetterChoose1.append(m)

# 获取匹配点的坐标
left_pts = np.float32([m.queryIdx for m in BetterChoose1]).reshape(-1, 1, 2)
right_pts = np.float32([m.trainIdx for m in BetterChoose1]).reshape(-1, 1, 2)

# 使用 RANSAC 方法估计变换矩阵
H, _ = cv2.findHomography(right_pts, left_pts, cv2.RANSAC, 5.0)

# 对右图进行透视变换
wrap = cv2.warpPerspective(R, H, (R.shape[1] + L.shape[1], R.shape[0] + R.shape[0]))

# 将左图与变换后的右图拼接
wrap[0:L.shape[0], 0:L.shape[1]] = L

# 显示拼接结果
cv2.imshow('Stitched Image', wrap)
cv2.waitKey(0)
cv2.destroyAllWindows()
