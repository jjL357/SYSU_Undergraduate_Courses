import cv2
import numpy as np
from skimage import feature

def compute_gradients(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return gradient_x, gradient_y

def compute_harris_response(gradient_x, gradient_y, alpha=0.04):
    Ix_Ix = gradient_x**2
    Ix_Iy = gradient_x * gradient_y
    Iy_Iy = gradient_y**2
    
    A = cv2.GaussianBlur(Ix_Ix, (5, 5), 0)
    B_C = cv2.GaussianBlur(Ix_Iy, (5, 5), 0)
    D = cv2.GaussianBlur(Iy_Iy, (5, 5), 0)
    
    detM = A * D - B_C**2
    traceM = A + D
    harris_response = detM - alpha * (traceM**2)
    return harris_response

def detect_corners(harris_response, coefficient=0.1, size=2):
    corners = []
    threshold = coefficient * harris_response.max()
    for y in range(size, harris_response.shape[0] - size):
        for x in range(size, harris_response.shape[1] - size):
            if harris_response[y, x] > threshold:
                neighborhood = harris_response[y - size:y + size + 1, x - size:x + size + 1]
                if harris_response[y, x] == neighborhood.max():
                    corners.append(cv2.KeyPoint(x, y, 1))
    return corners

def hog_descriptor(feature, keypoints, pixels_per_cell=8, orientations=9, cells_per_block=2):
    descriptor = []
    des_length = cells_per_block * cells_per_block * orientations
    for keypoint in keypoints:
        x, y = keypoint.pt
        cell_x = int(x // pixels_per_cell)
        cell_y = int(y // pixels_per_cell)
        begin_x = cell_x * pixels_per_cell
        begin_y = cell_y * pixels_per_cell
        begin = (cell_y * 75 + cell_x) * des_length
        if begin > 120000:
            begin = ((cell_y - 1) * 75 + (cell_x - 1)) * des_length
            begin_x = (cell_x - 1) * pixels_per_cell
            begin_y = (cell_y - 1) * pixels_per_cell
        desc = feature[begin:begin + des_length]
        if desc.shape[0] == des_length:
            descriptor.append(desc)
    return np.array(descriptor)


def match_descriptor_alternative(des_left, des_right, keypoints_left, keypoints_right, w):
    # 初始化匹配点列表
    good_matches = []
    
    # 计算左图中的每个关键点与右图中的所有关键点之间的距离，并寻找最近的匹配点
    for i, kp_left in enumerate(keypoints_left):
        if kp_left.pt[0] <= w // 2:  # 左图像的关键点在图像中心右侧则跳过
            continue

        min_distance = float('inf')  # 初始化最小距离
        best_match_index = -1  # 初始化最佳匹配点索引
        
        # 计算当前左图关键点与右图中所有关键点的距离
        for j, kp_right in enumerate(keypoints_right):
            if kp_right.pt[0] >= w // 2:  # 右图像的关键点在图像中心左侧则跳过
                continue
            
            # 计算描述子之间的欧式距离
            distance = np.linalg.norm(des_left[i] - des_right[j])

            # 更新最小距离和最佳匹配点索引
            if distance < min_distance:
                min_distance = distance
                best_match_index = j
        
        # 创建匹配点对象并添加到匹配点列表中
        match = cv2.DMatch()
        match.queryIdx = i
        match.trainIdx = best_match_index
        match.distance = min_distance
        good_matches.append(match)
    
    return good_matches

image1 = cv2.imread("D:\\picture\\uttower1.jpg")
image2 = cv2.imread("D:\\picture\\uttower2.jpg")
gradient_x_1, gradient_y_1 = compute_gradients(image1)
gradient_x_2, gradient_y_2 = compute_gradients(image2)

harris_response_1 = compute_harris_response(gradient_x_1, gradient_y_1)
harris_response_2 = compute_harris_response(gradient_x_2, gradient_y_2)

keypoints1 = detect_corners(harris_response_1)
keypoints2 = detect_corners(harris_response_2)
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
features1 = feature.hog(gray1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
features2 = feature.hog(gray2, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

descriptor1 = hog_descriptor(features1, keypoints1)
descriptor2 = hog_descriptor(features2, keypoints2)

descriptor1 = descriptor1.astype(np.float32)
descriptor2 = descriptor2.astype(np.float32)

good_matches = match_descriptor_alternative(descriptor1, descriptor2, keypoints1, keypoints2, image1.shape[1])

matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite("D:\\picture\\result\\uttower_matched_image_hog.png", matched_image)
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
h1, w1, _ = image1.shape
h2, w2, _ = image2.shape
corners = np.array([
    [0, 0],
    [0, h2],
    [w2, h2],
    [w2, 0]
], dtype=np.float32).reshape(-1, 1, 2)
transformed_corners = cv2.perspectiveTransform(corners, M).reshape(-1, 2)

all_corners = np.concatenate((transformed_corners, np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32)), axis=0)
min_x = min(all_corners[:, 0])
min_y = min(all_corners[:, 1])
max_x = max(all_corners[:, 0])
max_y = max(all_corners[:, 1])
shift = [-min_x, -min_y]
size = (int(max_x - min_x), int(max_y - min_y))

T = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])
M_final = np.dot(T, M)

stitched_image = cv2.warpPerspective(image1, M_final, size)
stitched_image[int(shift[1]):int(shift[1])+h2, int(shift[0]):int(shift[0])+w2] = image2

stitched_gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(stitched_gray, 1, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(contours[0])
stitched_image = stitched_image[y:y+h, x:x+w]

cv2.imwrite("D:\\picture\\result\\uttower_stitching_hog.png", stitched_image)
