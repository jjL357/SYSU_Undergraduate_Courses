import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature 
import scipy.ndimage
import skimage.util.shape
from skimage.feature import hog
def hog_descriptor(image, keypoints, winSize=(64, 64)):
    # 创建一个 HOG 描述符对象
    hog = cv2.HOGDescriptor(_winSize=winSize,
                            _blockSize=(winSize[0] // 2, winSize[1] // 2),
                            _blockStride=(winSize[0] // 4, winSize[1] // 4),
                            _cellSize=(winSize[0] // 8, winSize[1] // 8),
                            _nbins=9)
    descriptors = []  # 用于存储描述符
    valid_keypoints = []  # 用于存储有效的关键点

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])  # 提取关键点的坐标
        half_size = winSize[0] // 2  # 计算窗口大小的一半
        # 确保从图像中提取的补丁不会超出图像边界
        if x - half_size >= 0 and y - half_size >= 0 and x + half_size <= image.shape[1] and y + half_size <= image.shape[0]:
            # 提取关键点周围的图像补丁
            patch = image[y - half_size:y + half_size, x - half_size:x + half_size]
            # 使用 HOG 描述符对象计算补丁的描述符
            descriptor = hog.compute(patch)
            # 如果描述符不为空，则将其添加到列表中，并将关键点添加到有效关键点列表中
            if descriptor is not None:
                descriptors.append(descriptor)
                valid_keypoints.append(kp)

    # 将描述符列表转换为数组，并确保其形状合适
    if descriptors:
        return np.array(descriptors).squeeze(), valid_keypoints
    else:
        return np.array([]), []


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
    det_M = Ixx * Iyy - Ixy ** 2  # 行列式 = 特征值的乘积
    trace_M = Ixx + Iyy  # 迹 = 特征值的和
    harris_response = det_M - k * (trace_M ** 2)

    # 根据阈值找到角点并构建KeyPoint对象
    keypoints = []
    for i in range(harris_response.shape[0]):
        for j in range(harris_response.shape[1]):
            if harris_response[i, j] > threshold * harris_response.max():
                keypoints.append(cv2.KeyPoint(j, i, 1))
    
    return keypoints


def get_hog(image1, image2):

    
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # 转换为灰度用于检测和计算
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    keypoints1 = harris_corner_detection(gray1)
    keypoints2 = harris_corner_detection(gray2)
    
    descriptors1, keypoints1 = hog_descriptor(gray1, keypoints1)
    descriptors2, keypoints2 = hog_descriptor(gray2, keypoints2)
        
    # BFMatcher匹配,
    bf = cv2.BFMatcher()
    # matches = bf.match(descriptors1, descriptors2)
    # matches = sorted(matches, key = lambda x:x.distance)(不筛选效果不好)
    
    # 进行筛选
    matches_knn = bf.knnMatch(descriptors1, descriptors2, k=2)
    matches = []
    for m, n in matches_knn:
        if m.distance < 0.75 * n.distance:
            matches.append(m)
    
    # 绘制匹配结果
    image_match = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:], None, flags=2)
    return [keypoints1,keypoints2],[descriptors1,descriptors2],image_match,matches
    
    
# RANSAC 求解仿射变换矩阵
def ransac(keypoints1, keypoints2, matches, num_iterations=2000, tolerance=5.0,points_num = 200):
     # 仅使用指定数量的较好的匹配点进行 RANSAC
    matches = matches[:min(len(matches),points_num)]
    
    # 记录仿射变换矩阵
    matrix = None
    max_inliers = 0

    # 将关键点转换为点坐标
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    # 将 src_pts 转换为所需形状
    #print("src_pts shape:", src_pts.shape)
    #print("src_pts shape:", dst_pts.shape)
    
    for _ in range(num_iterations):
        # 随机选择 4 个点
        indices = np.random.choice(len(src_pts), 4, replace=False)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]
        
        M, _ = cv2.findHomography(src_sample, dst_sample)
        if M is None:
            continue
        projected_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), M)
        distances = np.linalg.norm(projected_pts.squeeze() - dst_pts, axis=1)

        # 统计在范围内的内点数量
        inliers = np.sum(distances < tolerance)
        # 更新最佳模型
        if inliers > max_inliers:
            max_inliers = inliers
            matrix = M
    
    return matrix


def stitch(image1, image2, matrix):
    # 获取图像1和图像2的高度和宽度
    h1, w1, _ = image1.shape
    h2, w2, _ = image2.shape
    
    # 定义图像2的四个角点
    corners = np.array([
        [0, 0],
        [0, h2],
        [w2, h2],
        [w2, 0]
    ], dtype=np.float32).reshape(-1, 1, 2)
    
    # 将角点通过仿射变换矩阵进行变换
    transformed_corners = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)

    # 将所有角点合并成一个数组，包括变换后的和图像1的四个角点
    all_corners = np.concatenate((transformed_corners, np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32)), axis=0)
    
    # 计算变换后图像的最小x、y和最大x、y
    min_x = min(all_corners[:, 0])
    min_y = min(all_corners[:, 1])
    max_x = max(all_corners[:, 0])
    max_y = max(all_corners[:, 1])
    
    # 计算平移量和新图像的尺寸
    shift = [-min_x, -min_y]
    size = (int(max_x - min_x), int(max_y - min_y))

    # 构造平移矩阵
    T = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])
    M_final = np.dot(T, matrix)

    # 通过仿射变换得到拼接后的图像
    stitched_image = cv2.warpPerspective(image1, M_final, size)
    stitched_image[int(shift[1]):int(shift[1])+h2, int(shift[0]):int(shift[0])+w2] = image2

    # 将拼接后的图像转换为灰度图，并进行阈值处理
    stitched_gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(stitched_gray, 1, 255, cv2.THRESH_BINARY)
    
    # 寻找图像中的最小外接矩形
    _,contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    
    # 裁剪图像
    stitched_image = stitched_image[y:y+h, x:x+w]
    return stitched_image

    
# 读取图片
image1 = cv2.imread('images\\uttower1.jpg')
image2 = cv2.imread('images\\uttower2.jpg')

keypoints_hog,descritor_hog,image_match_hog ,matches= get_hog(image1 ,image2)

cv2.imshow('hog Matched Image', image_match_hog)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite("results/uttower_match_hog.png",image_match_hog)
matrix = ransac(keypoints_hog[0],keypoints_hog[1],matches)

stitched_image = stitch(image1, image2, matrix)


cv2.imshow('Stitched Image', stitched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite("results/uttower_stitching_hog.png",stitched_image)


