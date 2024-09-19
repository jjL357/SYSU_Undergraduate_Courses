import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def sift_matching(img1, img2):
    # 创建SIFT对象
    sift = cv.SIFT_create()
    
    # 将图像转换为灰度图
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # 在两幅灰度图像上检测关键点和计算描述子
    keypoints1, desc1 = sift.detectAndCompute(img1_gray, None)
    keypoints2, desc2 = sift.detectAndCompute(img2_gray, None)

    # 使用FLANN进行特征匹配
    index_params = dict(algorithm=1, tree=3)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    result = flann.knnMatch(desc1, desc2, k=2)

    # 仅保留优质匹配点
    good_matches = []
    for i, (m, n) in enumerate(result):
        threshold = 0.5
        if m.distance / n.distance < threshold:
            good_matches.append(m)

    return keypoints1, keypoints2, good_matches

def ransac(keypoints1, keypoints2, matches):
    # RANSAC 参数
    num_iterations=2000
    tolerance=5.0
    best_model = None
    best_inliers = 0

    # 将关键点转换为点
    src_pts = np.float32([keypoints1[i.queryIdx].pt for i in matches])
    dst_pts = np.float32([keypoints2[i.trainIdx].pt for i in matches])

    # RANSAC 迭代
    for _ in range(num_iterations):
        # 随机选择4个点对应
        indices = np.random.choice(len(src_pts), 4, replace=False)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]

        # 计算该随机样本的单应性矩阵
        M, _ = cv.findHomography(src_sample, dst_sample)

        # 使用估计的单应性矩阵计算投影点
        projected_pts = cv.perspectiveTransform(src_pts.reshape(-1, 1, 2), M)

        # 计算投影点与实际目标点之间的欧几里得距离
        distances = np.linalg.norm(projected_pts.squeeze() - dst_pts, axis=1)

        # 统计在容差范围内的内点数量
        inliers = np.sum(distances < tolerance)

        # 如果发现更多内点，则更新最佳模型
        if inliers > best_inliers:
            best_inliers = inliers
            best_model = M

    return best_model



def match():
    # 定义边界
    (top, bottom, left, right) = (100, 100, 0, 500)
    
    # 读取图像
    img1 = cv.imread('images\\uttower1.jpg')
    img2 = cv.imread('images\\uttower2.jpg')

    # 给图像添加边界
    LeftImg = cv.copyMakeBorder(img1, top, bottom, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    RightImg = cv.copyMakeBorder(img2, top, bottom, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))

    # SIFT特征匹配
    keypoints1, keypoints2, matches = sift_matching(LeftImg, RightImg)
    
    # RANSAC算法估计单应性矩阵
    Matrix = ransac(keypoints1, keypoints2, matches)

    # 对右图进行透视变换
    rows, cols = LeftImg.shape[:2]
    warpImg = cv.warpPerspective(RightImg, np.array(Matrix), (RightImg.shape[1], RightImg.shape[0]), flags=cv.WARP_INVERSE_MAP)
    
    
    # 找到重叠区域
    for col in range(0, cols):
        if LeftImg[:, col].any() and warpImg[:, col].any():
            left = col
            break
    for col in range(cols-1, 0, -1):
        if LeftImg[:, col].any() and warpImg[:, col].any():
            right = col
            break

    # 图像拼接：尝试不同的混合策略
    res = np.zeros([rows, cols, 3], np.uint8)
    for row in range(0, rows):
        for col in range(0, cols):
            if not LeftImg[row, col].any():
                res[row, col] = warpImg[row, col]
            elif not warpImg[row, col].any():
                res[row, col] = LeftImg[row, col]
            else:
                LeftImgLen = float(abs(col - left))
                RightImgLen = float(abs(col - right))
                # 尝试使用不同的权重计算混合值
                if LeftImgLen < RightImgLen:
                    alpha = LeftImgLen / RightImgLen
                    res[row, col] = LeftImg[row, col] * (1-alpha) + warpImg[row, col] * alpha
                else:
                    alpha = RightImgLen / LeftImgLen
                    res[row, col] = LeftImg[row, col] * alpha + warpImg[row, col] * (1-alpha)

    # 显示和保存结果
    res = cv.cvtColor(res, cv.COLOR_BGR2RGB)
    plt.subplots(figsize=(12, 7))
    plt.axis('off')
    plt.imshow(res)
    #plt.savefig('Homework 1\\results\\uttower_stitching_sift.png')
    plt.show()


if __name__ == '__main__':
    match()
