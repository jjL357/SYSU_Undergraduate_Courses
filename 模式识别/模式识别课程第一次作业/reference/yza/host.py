import numpy as np
from skimage import filters
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
unpad = lambda x: x[:,:-1]

def harris_corner_detector(image, k=0.04):
    gray_image = image.copy()
    
    # Calculate the gradients using Sobel filters
    dx = np.gradient(gray_image, axis=1)
    dy = np.gradient(gray_image, axis=0)
    
    # Calculate the products of gradients
    dx2 = dx * dx
    dy2 = dy * dy
    dxy = dx * dy
    
    # Calculate the sums of gradients within a window
    window_size = 3
    sum_dx2 = convolve(dx2, np.ones((window_size, window_size)))
    sum_dy2 = convolve(dy2, np.ones((window_size, window_size)))
    sum_dxy = convolve(dxy, np.ones((window_size, window_size)))
    
    # Calculate the Harris response for each pixel
    det = sum_dx2 * sum_dy2 - sum_dxy**2
    trace = sum_dx2 + sum_dy2
    harris_response = det - k * trace**2
    corners = harris_response 
    
    return corners



import numpy as np

def ransac(pts1, pts2, correspondences, iterations=200, error_thresh=20):

    num_matches = correspondences.shape[0]
    sample_size = int(num_matches * 0.2)

    pts1_augmented = augment(pts1[correspondences[:, 0]])
    pts2_augmented = augment(pts2[correspondences[:, 1]])

    best_inliers = np.zeros(num_matches)
    max_inliers_count = 0

    for _ in range(iterations):
        current_inliers = np.zeros(num_matches, dtype=np.int32)
        sample_indices = np.random.choice(num_matches, sample_size, replace=False)
        sample_pts1 = pts1_augmented[sample_indices]
        sample_pts2 = pts2_augmented[sample_indices]

        affine_matrix, _, _, _ = np.linalg.lstsq(sample_pts2, sample_pts1, rcond=None)
        affine_matrix[:, 2] = [0, 0, 1]  # Enforcing an affine transformation

        reprojected = pts2_augmented @ affine_matrix
        errors = np.linalg.norm(reprojected - pts1_augmented, axis=1) ** 2
        current_inliers = errors < error_thresh
        current_inlier_count = np.sum(current_inliers)

        if current_inlier_count > max_inliers_count:
            best_inliers = current_inliers.copy()
            max_inliers_count = current_inlier_count

    final_affine_matrix, _, _, _ = np.linalg.lstsq(pts2_augmented[best_inliers], pts1_augmented[best_inliers], rcond=None)
    final_affine_matrix[:, 2] = [0, 0, 1]

    return final_affine_matrix, correspondences[best_inliers]

def augment(points):
   
    return np.hstack((points, np.ones((points.shape[0], 1))))


import numpy as np

def extract_descriptors(img, points, feature_extractor, window_size=16):
   
    # Ensure the image is in the correct floating point format
    img = img.astype(np.float32)
    descriptors = []

    # Compute the half size of the window to simplify boundary calculations
    half_window = window_size // 2

    for point in points:
        y, x = point
        # Extract the patch centered at the keypoint
        patch = img[max(0, y - half_window): y + (window_size+1)//2,
                    max(0, x - half_window): x + (window_size+1)//2]
        # Compute the descriptor using the provided function
        descriptors.append(feature_extractor(patch))

    return np.array(descriptors)



from scipy.ndimage import affine_transform
def warp_image(img, H, output_shape, offset):

    # Note about affine_transfomr function:
    # Given an output image pixel index vector o,
    # the pixel value is determined from the input image at position
    # np.dot(matrix,o) + offset.
    Hinv = np.linalg.inv(H)
    m = Hinv.T[:2,:2]
    b = Hinv.T[:2,2]
    img_warped = affine_transform(img.astype(np.float32),
                                  m, b+offset,
                                  output_shape,
                                  cval=-1)

    return img_warped


def get_output_space(img_ref, imgs, transforms):
   

    assert (len(imgs) == len(transforms))
    print(img_ref.shape)
    r, c= img_ref.shape
    corners = np.array([[0, 0], [r, 0], [0, c], [r, c]])
    all_corners = [corners]

    for i in range(len(imgs)):
        r, c = imgs[i].shape
        H = transforms[i]
        corners = np.array([[0, 0], [r, 0], [0, c], [r, c]])
        warped_corners = corners.dot(H[:2,:2]) + H[2,:2]
        all_corners.append(warped_corners)

    # Find the extents of both the reference image and the warped
    # target image
    all_corners = np.vstack(all_corners)

    # The overall output shape will be max - min
    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    output_shape = (corner_max - corner_min)

    # Ensure integer shape with np.ceil and dtype conversion
    output_shape = np.ceil(output_shape).astype(int)
    offset = corner_min

    return output_shape, offset




import numpy as np
from scipy.spatial.distance import cdist

def find_feature_matches(features1, features2, match_ratio=0.5):
   
    # Calculate pairwise Euclidean distances between feature descriptors
    pairwise_distances = cdist(features1, features2)
    match_indices = []

    for index, distances in enumerate(pairwise_distances):
        sorted_indices = np.argsort(distances)[:2]  # Get indices of the two smallest distances
        if distances[sorted_indices[0]] / distances[sorted_indices[1]] < match_ratio:
            match_indices.append([index, sorted_indices[0]])

    return np.array(match_indices)


import numpy as np
from skimage import filters
from skimage.util import view_as_blocks

def calculate_hog_features(image_patch, cell_size=(8, 8)):
   
    # Preconditions to ensure the patch can be divided evenly into cells
    assert image_patch.shape[0] % cell_size[0] == 0, "Patch height must be divisible by cell height."
    assert image_patch.shape[1] % cell_size[1] == 0, "Patch width must be divisible by cell width."

    # Configuration for histogram
    number_of_bins = 9
    bin_range = 180 // number_of_bins

    # Compute gradients using Sobel filters
    gradient_x = filters.sobel_v(image_patch)
    gradient_y = filters.sobel_h(image_patch)

    # Compute the magnitude and orientation of the gradient
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    orientation = np.mod(np.degrees(np.arctan2(gradient_y, gradient_x)), 180)

    # View the gradients as blocks (cells)
    mag_cells = view_as_blocks(magnitude, block_shape=cell_size)
    ori_cells = view_as_blocks(orientation, block_shape=cell_size)
    cell_rows, cell_cols = mag_cells.shape[0], mag_cells.shape[1]

    histograms = np.zeros((cell_rows, cell_cols, number_of_bins))

    # Loop over each cell
    for r in range(cell_rows):
        for c in range(cell_cols):
            # Accumulate votes in histogram bins
            for i in range(cell_size[0]):
                for j in range(cell_size[1]):
                    bin_index = int(ori_cells[r, c, i, j] // bin_range)
                    if bin_index == 9:  # Ensure the maximum bin index is 8
                        bin_index = 8
                    histograms[r, c, bin_index] += mag_cells[r, c, i, j]

    # Normalize the histograms
    normalized_histograms = (histograms - histograms.mean(axis=(0, 1))) / histograms.std(axis=(0, 1))
    feature_vector = normalized_histograms.ravel()  # Flatten to create a feature vector

    return feature_vector

