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
    image_match = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=2)
    return [keypoints1,keypoints2],[descriptors1,descriptors2],image_match,matches

# RANSAC 求解仿射变换矩阵
def ransac(keypoints1, keypoints2, matches, num_iterations=5000, tolerance=5.0):
    
    # 仅使用指定数量的较好的匹配点进行 RANSAC(效果不好)
    # matches = matches[:min(len(matches),points_num)]
    
    # 记录仿射变换矩阵
    matrix = None
    max_inliers = 0

    # 将关键点转换为点坐标
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # 直接调库(直接调库效果稳定)
    # matrix ,_= cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # return matrix
    
    # 以下手动实现出现了有可能每次运行,其中会有几次效果不好的情况,效果不稳定
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
    h1, w1, _ = image1.shape
    h2, w2, _ = image2.shape
    corners = np.array([
        [0, 0],
        [0, h2],
        [w2, h2],
        [w2, 0]
    ], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)

    all_corners = np.concatenate((transformed_corners, np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32)), axis=0)
    min_x = min(all_corners[:, 0])
    min_y = min(all_corners[:, 1])
    max_x = max(all_corners[:, 0])
    max_y = max(all_corners[:, 1])
    shift = [-min_x, -min_y]
    size = (int(max_x - min_x), int(max_y - min_y))

    T = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])
    M_final = np.dot(T, matrix)

    stitched_image = cv2.warpPerspective(image1, M_final, size)
    stitched_image[int(shift[1]):int(shift[1])+h2, int(shift[0]):int(shift[0])+w2] = image2

    stitched_gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(stitched_gray, 1, 255, cv2.THRESH_BINARY)
    _,contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    stitched_image = stitched_image[y:y+h, x:x+w]
    return stitched_image


def resize_image(image1, image2): # 将image2 resize 和 image2 同等大小
    # 获取图像1的尺寸
    h1, w1, _ = image1.shape
    
    # 调整图像2的大小为图像1的大小
    resized_image2 = cv2.resize(image2, (w1, h1))
    
    return resized_image2    

img1 = cv2.imread('images\\yosemite1.jpg')          
img2 = cv2.imread('images\\yosemite2.jpg')          
img3 = cv2.imread('images\\yosemite3.jpg')          
img4 = cv2.imread('images\\yosemite4.jpg')          

keypoints_sift,descritor_sift,image_match_sift ,matches= get_sift(img1 ,img2)
matrix = ransac(keypoints_sift[0],keypoints_sift[1],matches)
stitched_image1 = stitch(img1, img2, matrix)

stitched_image1 = resize_image(img3,stitched_image1)
keypoints_sift,descritor_sift,image_match_sift ,matches= get_sift(stitched_image1 ,img3)
matrix = ransac(keypoints_sift[0],keypoints_sift[1],matches)
stitched_image2 = stitch(stitched_image1 ,img3, matrix)

stitched_image2 = resize_image(img4,stitched_image2)
keypoints_sift,descritor_sift,image_match_sift ,matches= get_sift(stitched_image2 ,img4)
matrix = ransac(keypoints_sift[0],keypoints_sift[1],matches)
stitched_image = stitch(stitched_image2, img4, matrix)

cv2.imshow('Stitche', stitched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("results/yosemite_stitching.png",stitched_image)