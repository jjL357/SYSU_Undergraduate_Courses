import cv2
import numpy as np

def stitch_image(image1, image2):
    sift = cv2.SIFT_create()
    # 计算图像1和图像2的关键点和描述子
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # 使用 BFMatcher 进行特征匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 使用 Lowe's ratio test 筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    # 绘制匹配结果
    
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 保存匹配结果
    # cv2.imwrite('results/uttower_match.png', matched_image)
    # 提取好的匹配点的位置
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 使用RANSAC找到仿射变换矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 获取图像尺寸和进行仿射变换
    h1, w1, _ = image1.shape
    h2, w2, _ = image2.shape
    corners = np.array([
        [0, 0],
        [0, h2],
        [w2, h2],
        [w2, 0]
    ], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, M).reshape(-1, 2)

    # 找到包含两图的最小区域
    all_corners = np.concatenate((transformed_corners, np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32)), axis=0)
    min_x = min(all_corners[:, 0])
    min_y = min(all_corners[:, 1])
    max_x = max(all_corners[:, 0])
    max_y = max(all_corners[:, 1])
    shift = [-min_x, -min_y]
    size = (int(max_x - min_x), int(max_y - min_y))

    # 创建平移矩阵并结合仿射矩阵
    T = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])
    M_final = np.dot(T, M)

    # 进行透视变换，拼接图像
    stitched_image = cv2.warpPerspective(image1, M_final, size)
    stitched_image[int(shift[1]):int(shift[1])+h2, int(shift[0]):int(shift[0])+w2] = image2

    # 去除黑边
    stitched_gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(stitched_gray, 1, 255, cv2.THRESH_BINARY)
    _,contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    stitched_image = stitched_image[y:y+h, x:x+w]
    return stitched_image

def resize_image(image, width=None, height=None):
    """
    Resize the image to the specified width and height.
    """
    if width is None and height is None:
        return image
    if width is None:
        ratio = height / image.shape[0]
        width = int(image.shape[1] * ratio)
    elif height is None:
        ratio = width / image.shape[1]
        height = int(image.shape[0] * ratio)
    resized_image = cv2.resize(image, (width, height))
    return resized_image

# 读取图像
image1 = cv2.imread("images\\yosemite1.jpg")
image2 = cv2.imread("images\\yosemite2.jpg")
image3 = cv2.imread("images\\yosemite3.jpg")
image4 = cv2.imread("images\\yosemite4.jpg")

# 拼接第一对图像
stitched_image1 = stitch_image(image1, image2)
cv2.imwrite("D:\\picture\\result\\uttower_stitching_sift_1.png", stitched_image1)

# 拼接第二对图像
# stitched_image2 = stitch_image(image3, image4)
# cv2.imwrite("D:\\picture\\result\\uttower_stitching_sift_2.png", stitched_image2)

# # 调整图像尺寸使其与stitched_image1相同
# stitched_image1 = resize_image(stitched_image1, width=stitched_image2.shape[1], height=stitched_image2.shape[0])
# # 拼接最终图像
# stitched_image_final = stitch_image(stitched_image1, stitched_image2)
# 
stitched_image1 = resize_image(stitched_image1, width=image3.shape[1], height=image3.shape[0])
stitched_image2 = stitch_image(stitched_image1, image3)

stitched_image2 = resize_image(stitched_image2, width=image4.shape[1], height=image4.shape[0])
stitched_image_final = stitch_image(stitched_image2, image4)

cv2.imshow('Stitched Image', stitched_image_final)
cv2.waitKey(0)
cv2.destroyAllWindows()