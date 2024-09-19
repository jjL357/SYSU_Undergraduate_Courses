import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import skimage.feature
import scipy.ndimage
import skimage.util.shape

# Harris 角点检测
def harris_corners(image, threshold=0.01, window_size=3, k=0.04):
    # 1. 计算图像的梯度
    dx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    dy = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)

    # 2. 计算 Harris 角点响应函数
    Ixx = dx ** 2
    Ixy = dx * dy
    Iyy = dy ** 2

    height, width = image.shape
    offset = window_size // 2
    corner_response = np.zeros((height, width))

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            window_Ixx = Ixx[y - offset : y + offset + 1, x - offset : x + offset + 1]
            window_Ixy = Ixy[y - offset : y + offset + 1, x - offset : x + offset + 1]
            window_Iyy = Iyy[y - offset : y + offset + 1, x - offset : x + offset + 1]

            # 计算局部窗口内的梯度协方差矩阵的特征值
            Sxx = np.sum(window_Ixx)
            Sxy = np.sum(window_Ixy)
            Syy = np.sum(window_Iyy)

            # 计算角点响应函数值
            det = Sxx * Syy - Sxy ** 2
            trace = Sxx + Syy
            corner_response[y, x] = det - k * trace ** 2

    # 3. 对角点响应函数进行阈值处理
    corner_response[corner_response < threshold * np.max(corner_response)] = 0

    # 4. 非极大值抑制
    corner_response = cv.dilate(corner_response, None)

    return corner_response

# 使用 RANSAC 算法估计两组关键点之间的单应性矩阵
def estimate_homography_ransac(keypoints_img1, keypoints_img2, matches, n_iters=100, threshold=15):

    # 添加一列全1，方便计算单应性矩阵时转换为齐次坐标
    def add_ones_column(x):
        return np.hstack([x, np.ones((x.shape[0], 1))])

    N = matches.shape[0]
    n_samples = int(N * 0.25)
    matched1 = add_ones_column(keypoints_img1[matches[:,0]])
    matched2 = add_ones_column(keypoints_img2[matches[:,1]])
    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC 迭代
    for _ in range(n_iters):
        np.random.shuffle(matches)
        samples = matches[:n_samples]
        points_img1 = add_ones_column(keypoints_img1[samples[:,0]])
        points_img2 = add_ones_column(keypoints_img2[samples[:,1]])
        H, _ = np.linalg.lstsq(points_img2, points_img1, rcond=None)[:2]
        transformed_points = np.dot(matched2, H)
        # 计算内点
        inliers_mask = np.linalg.norm(transformed_points - matched1, axis=1)**2 < threshold
        inliers_count = np.sum(inliers_mask)
        # 更新最大内点记录
        if inliers_count > n_inliers:
            max_inliers = inliers_mask.copy()
            n_inliers = inliers_count

    # 使用所有内点重新计算单应性矩阵 H
    H, _ = np.linalg.lstsq(matched2[max_inliers], matched1[max_inliers], rcond=None)[:2]
    return H, matches[max_inliers]

# 对图像进行仿射变换
def warp_image(img, H, output_shape, offset):    
    H_inv = np.linalg.inv(H)
    m = H_inv.T[:2, :2]
    b = H_inv.T[:2, 2] + offset
    img_warped = scipy.ndimage.affine_transform(img.astype(np.float32), m, b, output_shape, cval=-1)
    return img_warped

# 计算图像块的 HOG 特征描述子
def calculate_hog_descriptor(patch, cell_size=(8, 8),n_bins = 9, degrees_per_bin = 20):

    # 计算梯度
    gradient_x = cv.Sobel(patch, cv.CV_64F, 1, 0, ksize=3)
    gradient_y = cv.Sobel(patch, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    angle = (np.arctan2(gradient_y, gradient_x) * 180 / np.pi) % 180

    # 将图像切割为 cells
    hog_cells = skimage.util.shape.view_as_blocks(magnitude, cell_size)
    angle_cells = skimage.util.shape.view_as_blocks(angle, cell_size)

    rows, cols = hog_cells.shape[:2]
    cells = np.zeros((rows, cols, n_bins))

    # 计算每个 cell 的直方图
    for row in range(rows):
        for col in range(cols):
            for y in range(cell_size[0]):
                for x in range(cell_size[1]):
                    angle_value = angle_cells[row, col, y, x]
                    bin_index = int(angle_value) // degrees_per_bin
                    if(bin_index >= 9):
                        bin_index = 0
                    cells[row, col, bin_index] += hog_cells[row, col, y, x]

    # 归一化
    cells = (cells - cells.mean()) / (cells.std())
    hog_descriptor = cells.reshape(-1)

    return hog_descriptor

# 计算关键点处图像块的 HOG 特征描述子
def calculate_keypoints_hog_descriptor(image, keypoints, patch_size=8):
    image = image.astype(np.float32)
    descriptors = []
    for keypoint in keypoints:
        y, x = keypoint
        half_patch_size = patch_size // 2
        begin_x = max(0, x - half_patch_size)
        end_x = min(image.shape[1], x + half_patch_size)
        begin_y = max(0, y - half_patch_size)
        end_y = min(image.shape[0], y + half_patch_size)
        descriptors.append(calculate_hog_descriptor(image[begin_y:end_y, begin_x:end_x]))

    return np.array(descriptors)

# 计算两组点集之间的欧氏距离
def calculate_euclidean_distance(desc1, desc2):
    (n, m) = (desc1.shape[0], desc2.shape[0])
    distances = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            distances[i, j] = np.sqrt(np.sum((desc1[i] - desc2[j]) ** 2))

    return distances

# 选择匹配的描述子对
def select_good_matches(desc1, desc2, threshold=0.75):
    pairs = []
    N = desc1.shape[0]
    distances = calculate_euclidean_distance(desc1, desc2)

    for i in range(N):
        sorted_distances = np.sort(distances[i,])
        # 判断是否匹配
        if sorted_distances[0] < sorted_distances[1] * threshold:
            pairs.append([i, np.argmin(distances[i, :])])

    return np.asarray(pairs)

# 读取图像并进行 Harris 角点检测
def read_image_and_detect_corners(input_path):
    image = cv.imread(input_path,0)
    response = harris_corners(image)
    corners = skimage.feature.corner_peaks(response, threshold_rel=0.05, exclude_border=8)
    return image, corners

# 绘制匹配结果
def draw_matching_results(image1, image2, keypoints1, keypoints2, pairs):
    output_path = 'Homework 1\\results\\uttower_match_hog.png'
    plt.subplots(figsize=(12, 7))
    image = np.concatenate([image1, image2], axis=1)
    offset = image1.shape
    plt.imshow(image, cmap='gray')
    
    for i in range(pairs.shape[0]):
        idx1 = pairs[i, 0]
        idx2 = pairs[i, 1]
        plt.plot((keypoints1[idx1, 1], keypoints2[idx2, 1] + offset[1]),
                 (keypoints1[idx1, 0], keypoints2[idx2, 0]), '-', color='purple')
        plt.plot(keypoints1[idx1, 1], keypoints1[idx1, 0], 'ro', markersize=5)
        plt.plot(keypoints2[idx2, 1] + offset[1], keypoints2[idx2, 0], 'ro', markersize=5)
    
    plt.axis('off')
    plt.savefig(output_path)
    plt.show()

# 主函数
def main(image1, image2, keypoints1, keypoints2):
    # 计算图像关键点的 HOG 描述子
    descriptors_img1 = calculate_keypoints_hog_descriptor(image1, keypoints1, patch_size=16)
    descriptors_img2 = calculate_keypoints_hog_descriptor(image2, keypoints2, patch_size=16)
    
    # 选择好的匹配对
    matches = select_good_matches(descriptors_img1, descriptors_img2, 0.63)
    
    # 使用 RANSAC 算法估计两组关键点之间的单应性矩阵
    homography_matrix, inliers = estimate_homography_ransac(keypoints1, keypoints2, matches, threshold=5)
    
    # 绘制匹配结果
    draw_matching_results(image1, image2, keypoints1, keypoints2, inliers)
    
    # 计算变换后图像的尺寸和偏移量
    rows1, cols1 = image1.shape
    corners1 = np.array([[0, 0], [rows1, 0], [0, cols1], [rows1, cols1]])
    rows2, cols2 = image2.shape
    corners2 = np.array([[0, 0], [rows2, 0], [0, cols2], [rows2, cols2]])
    all_corners = np.vstack([corners1, corners2.dot(homography_matrix[:2, :2]) + homography_matrix[2, :2]])
    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    output_shape = np.ceil(corner_max - corner_min).astype(int)
    offset = corner_min
    
    # 对图像进行仿射变换
    image1_warped = warp_image(image1, np.eye(3), output_shape, offset)
    image2_warped = warp_image(image2, homography_matrix, output_shape, offset)
    
    # 创建图像掩码
    image1_mask = (image1_warped != -1)
    image2_mask = (image2_warped != -1)
    
    # 合并变换后的图像
    merged = image1_warped + image2_warped
    overlap = (image1_mask * 1.0 + image2_mask)
    stitched_image = merged / np.maximum(overlap, 1)
    
    # 绘制拼接后的图像
    plt.subplots(figsize=(12, 7))
    plt.imshow(stitched_image, cmap='gray')
    plt.axis('off')
    
    # 保存拼接后的图像
    plt.savefig('Homework 1\\results\\uttower_stitching_hog.png')
    plt.show()


if __name__ == '__main__':
    image1, keypoints1 = read_image_and_detect_corners('Homework 1\\images\\uttower1.jpg')
    image2, keypoints2 = read_image_and_detect_corners('Homework 1\\images\\uttower2.jpg')
    main(image1, image2, keypoints1, keypoints2)
