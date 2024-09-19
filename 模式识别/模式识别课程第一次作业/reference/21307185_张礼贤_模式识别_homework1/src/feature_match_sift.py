import cv2
import numpy as np

def sift_feature_matching(img1, img2):
    # 初始化 SIFT 特征提取器
    sift = cv2.SIFT_create()

    # 检测关键点并计算描述子
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 创建暴力匹配器
    bf = cv2.BFMatcher()

    # 使用欧几里得距离进行特征匹配
    matches = bf.knnMatch(des1, des2, k=2)

    # 筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 绘制匹配结果
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 保存匹配结果图像
    cv2.imwrite('Homework 1\\results\\uttower_match_sift.png', img_matches)

def draw_matches(img1, kp1, img2, kp2, matches):
    # 将关键点列表转换为 list 类型
    kp1_list = [cv2.KeyPoint(x[0][0], x[0][1], _size=30) for x in kp1]
    kp2_list = [cv2.KeyPoint(x[0][0], x[0][1], _size=30) for x in kp2]

    # 绘制匹配结果
    img_matches = cv2.drawMatches(img1, kp1_list, img2, kp2_list, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches

# 读取两幅图像
img1 = cv2.imread('Homework 1\\images\\uttower1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('Homework 1\\images\\uttower2.jpg', cv2.IMREAD_GRAYSCALE)

# 使用 SIFT 特征进行匹配
sift_feature_matching(img1, img2)
