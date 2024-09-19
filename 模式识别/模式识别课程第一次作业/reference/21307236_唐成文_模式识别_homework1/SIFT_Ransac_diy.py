import cv2
import numpy as np

def ransac_homography(src_pts, dst_pts, max_iter=1000, tolerance=5.0):
    best_inliers = []
    best_homography = None

    for _ in range(max_iter):
        # Randomly select 4 points
        indices = np.random.choice(len(src_pts), 4, replace=False)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]

        # Compute homography
        homography, _ = cv2.findHomography(src_sample, dst_sample)

        # Apply homography to all points
        transformed_pts = cv2.perspectiveTransform(src_pts, homography)

        # Calculate Euclidean distances between transformed points and destination points
        distances = np.sqrt(np.sum((transformed_pts - dst_pts) ** 2, axis=2))

        # Count inliers (points with distances < tolerance)
        inliers = np.sum(distances < tolerance, axis=1)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_homography = homography

    # Refit homography using all inliers
    src_inliers = src_pts[best_inliers > 0]
    dst_inliers = dst_pts[best_inliers > 0]
    best_homography, _ = cv2.findHomography(src_inliers, dst_inliers)

    return best_homography

image1 = cv2.imread("D:\\picture\\uttower1.jpg")
image2 = cv2.imread("D:\\picture\\uttower2.jpg")
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

M = ransac_homography(src_pts, dst_pts)

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

cv2.imwrite("D:\\picture\\result\\uttower_stitching_sift_ransac_diy.png", stitched_image)
