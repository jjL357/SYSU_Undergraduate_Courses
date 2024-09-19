import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature
import scipy.ndimage
import skimage.util.shape
def get_sift(image1,image2):

    #创建sift检测器
    sift = cv2.SIFT_create()

    # 计算特征点和sift描述子
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # BFMatcher匹配,使用欧几里得距离作为特征之间相似度的度量
    bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key = lambda x:x.distance)

    # 绘制匹配结果
    image_match = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:200], None, flags=2)
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

    for _ in range(num_iterations):
        # 随机选择 4 个点
        indices = np.random.choice(len(src_pts), 4, replace=False)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]

        M, _ = cv2.findHomography(src_sample, dst_sample)
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
    # 获取图像尺寸
    h1, w1, _ = image1.shape
    h2, w2, _ = image2.shape
    
    # 计算图像1的四个角在图像2上的映射位置
    corners = np.array([
        [0, 0],
        [0, h2],
        [w2, h2],
        [w2, 0]
    ], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)

    # 计算包含两个图像所有角点的最小矩形框
    all_corners = np.concatenate((transformed_corners, np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32)), axis=0)
    min_x = min(all_corners[:, 0])
    min_y = min(all_corners[:, 1])
    max_x = max(all_corners[:, 0])
    max_y = max(all_corners[:, 1])
    
    # 计算平移和大小
    shift = [-min_x, -min_y]
    size = (int(max_x - min_x), int(max_y - min_y))

    # 构建最终的变换矩阵
    T = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])
    M_final = np.dot(T, matrix)

    # 对图像1进行透视变换
    stitched_image = cv2.warpPerspective(image1, M_final, size)
    
    # 将图像2叠加在透视变换后的图像1上
    stitched_image[int(shift[1]):int(shift[1])+h2, int(shift[0]):int(shift[0])+w2] = image2

    # 将图像转换为灰度图，并进行阈值处理
    stitched_gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(stitched_gray, 1, 255, cv2.THRESH_BINARY)
    
    # 寻找轮廓并裁剪图像
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    stitched_image = stitched_image[y:y+h, x:x+w]
    
    return stitched_image


    
# 读取图片
image1 = cv2.imread('images\\uttower1.jpg')
image2 = cv2.imread('images\\uttower2.jpg')

keypoints_sift,descritor_sift,image_match_sift ,matches= get_sift(image1 ,image2)

matrix = ransac(keypoints_sift[0],keypoints_sift[1],matches)

stitched_image = stitch(image1, image2, matrix)

cv2.imshow('SIFT Matched Image', image_match_sift)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite("results/uttower_match_sift.png",image_match_sift)
cv2.imshow('Stitched Image', stitched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite("results/uttower_stitching_sift.png",stitched_image)


