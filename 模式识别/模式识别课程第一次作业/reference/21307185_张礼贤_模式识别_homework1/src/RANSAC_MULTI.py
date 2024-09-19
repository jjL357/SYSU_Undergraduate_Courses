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


def match(img1, img2, num):
    # 定义边界
    (top, bottom, left, right) = (100, 100, 0, 500)

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
    if(num == 3):
        res = remove_black_rectangles(res)
    plt.subplots(figsize=(12, 7))
    plt.axis('off')
    plt.imshow(res)

    plt.savefig(f'Homework 1\\results\\yosemite_stitching_sift_{num}.png')
    plt.show()

def remove_black_rectangles(img):
    # 加载图片  

    # 创建一个与原始图像大小相同的全零数组作为掩码  
    mask = np.zeros(img.shape[:2], dtype=np.uint8)  

    # 遍历图像的所有像素，查找纯黑色像素（0, 0, 0）  
    # 注意：这种方法在图像很大时可能非常慢  
    # 为了优化，可以考虑使用cv2.inRange()或其他方法  
    for y in range(img.shape[0]):  
        for x in range(img.shape[1]):  
            if np.all(img[y, x] == [0, 0, 0]):  
                mask[y, x] = 255  # 标记为白色（掩码中的前景）  

    # 使用掩码来填充或删除黑色区域  
    # 这里我们选择填充黑色区域为白色（或其他颜色）  
    # 你可以使用0来填充为黑色，但通常我们想要删除它们  
    # 假设我们想要用白色替换它们  
    img_without_black = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)  
    # 如果你只是想用黑色填充它们（实际上就是删除它们），可以这样做：  
    # img_without_black = np.copy(img)  
    # img_without_black[mask == 255] = [0, 0, 0]  

    return img_without_black

if __name__ == '__main__':
    img1 = cv.imread('Homework 1\\images\\yosemite1.jpg')
    img2 = cv.imread('Homework 1\\images\\yosemite2.jpg')
    match(img1, img2, 1)

    img1 = cv.imread('Homework 1\\images\\yosemite3.jpg')
    img2 = cv.imread('Homework 1\\images\\yosemite4.jpg')
    match(img1, img2, 2)

    img1 = cv.imread('Homework 1\\results\\yosemite_stitching_sift_1.png')
    img2 = cv.imread('Homework 1\\results\\yosemite_stitching_sift_2.png')
    match(img1, img2, 3)

