import cv2
import numpy as np
import matplotlib.pyplot as plt

def hog_descriptor(image, keypoints, winSize=(64, 64)):
    hog = cv2.HOGDescriptor(_winSize=winSize,
                            _blockSize=(winSize[0] // 2, winSize[1] // 2),
                            _blockStride=(winSize[0] // 4, winSize[1] // 4),
                            _cellSize=(winSize[0] // 8, winSize[1] // 8),
                            _nbins=9)
    descriptors = []
    valid_keypoints = []

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        half_size = winSize[0] // 2
        if x - half_size >= 0 and y - half_size >= 0 and x + half_size <= image.shape[1] and y + half_size <= image.shape[0]:
            patch = image[y - half_size:y + half_size, x - half_size:x + half_size]
            descriptor = hog.compute(patch)
            if descriptor is not None:
                descriptors.append(descriptor)
                valid_keypoints.append(kp)

    if descriptors:
        return np.array(descriptors).squeeze(), valid_keypoints
    else:
        return np.array([]), []

def match_keypoints(img1, img2, feature_type='SIFT'):
    # 读取图像
    image1 = cv2.imread(img1)  # 使用彩色图像
    image2 = cv2.imread(img2)  # 使用彩色图像

    # 确保图像成功加载
    if image1 is None or image2 is None:
        raise ValueError("One of the images didn't load. Check the file paths.")

    descriptor = cv2.SIFT_create()  # 使用 SIFT 检测关键点，然后计算 HOG 描述符

    # 特征检测与描述子计算
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # 转换为灰度用于检测和计算
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    
    keypoints1, descriptors1 = descriptor.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = descriptor.detectAndCompute(gray2, None)
    if feature_type == 'HOG':
        descriptors1, keypoints1 = hog_descriptor(gray1, keypoints1)
        descriptors2, keypoints2 = hog_descriptor(gray2, keypoints2)
        if not descriptors1.size or not descriptors2.size:
            raise ValueError("Insufficient valid keypoints for matching.")

    # 描述子匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 绘制匹配结果
    match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=2)

    # RANSAC 求解仿射变换矩阵
    if len(matches) >= 4:
        src_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 计算第二幅图像变换到第一幅图像坐标系下的新边界
        h2, w2, _ = image2.shape
        corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, matrix)
        
        # 计算拼接图像的新边界
        h1, w1, _ = image1.shape
        all_corners = np.concatenate((transformed_corners, np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)), axis=0)
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
        
        
        #创建一个黑布
        stitched_img = np.zeros((y_max-y_min,x_max-x_min , 3), dtype=np.uint8)
        
        # 将第一幅图像复制到结果图像中
        stitched_img[translation_dist[1]:translation_dist[1] + h1, translation_dist[0]:translation_dist[0] + w1] = image1
        
        # 应用变换，将第二幅图变换到第一幅图坐标系下，并将其结果合并到拼接图中
        # 使用更好的插值方法
        # 使用双线性插值变换第二幅图像，减少锯齿
        transformed_img2 = cv2.warpPerspective(image2, H_translation.dot(matrix), (x_max-x_min, y_max-y_min), flags=cv2.INTER_LINEAR)

        # 如果transformed_img2是彩色图像，则取其灰度图作为掩膜
        if transformed_img2.ndim == 3:
            gray_transformed = cv2.cvtColor(transformed_img2, cv2.COLOR_BGR2GRAY)
        else:
            gray_transformed = transformed_img2.copy()

        # 创建一个二值掩膜，表示变换后图像的非零像素区域
        binary_mask = gray_transformed > 20

        # 对二值掩膜应用高斯模糊，产生平滑的边界
        soft_mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (5, 5), 0)

        # 将模糊掩膜转换为二进制掩膜
        binary_mask = (soft_mask > 0.5).astype(np.uint8)

        # 创建一个反掩膜，对于stitched_img中不应该由transformed_img2覆盖的区域
        inverse_mask = 1 - binary_mask

        # 更新stitched_img中的像素，只在反掩膜的区域内
        stitched_img = cv2.bitwise_and(stitched_img, stitched_img, mask=inverse_mask)

        # 添加变换后的图像到stitched_img中，只在掩膜的区域内
        stitched_img += cv2.bitwise_or(stitched_img, transformed_img2, mask=binary_mask)


    return match_img, stitched_img

# 使用 SIFT 特征进行匹配和拼接
match_img_sift, stitched_img_sift = match_keypoints('images\\uttower1.jpg', 'images\\uttower2.jpg', 'SIFT')
cv2.imshow('SIFT Matched Image', match_img_sift)
cv2.imshow('SIFT Stitched Image', stitched_img_sift)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 使用 HOG 特征进行匹配和拼接
match_img_sift, stitched_img_sift = match_keypoints('images\\uttower1.jpg', 'images\\uttower2.jpg', 'HOG')
cv2.imshow('HOG Matched Image', match_img_sift)
cv2.imshow('HOG Stitched Image', stitched_img_sift)
cv2.waitKey(0)
cv2.destroyAllWindows()