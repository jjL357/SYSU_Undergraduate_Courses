import cv2
import numpy as np

def compute_gradients(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gradient_x=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    gradient_y=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    return gradient_x, gradient_y

def compute_harris_response(gradient_x,gradient_y,alpha=0.04):
    Ix_Ix=gradient_x**2
    Ix_Iy=gradient_x*gradient_y
    Iy_Iy=gradient_y**2

    A=cv2.GaussianBlur(Ix_Ix,(5, 5),0)
    B_C=cv2.GaussianBlur(Ix_Iy,(5, 5),0)
    D=cv2.GaussianBlur(Iy_Iy,(5, 5),0)
    
    detM=A*D-B_C**2
    traceM = A + D
    harris_response=detM-alpha*(traceM**2)
    return harris_response

def detect_corners(harris_response,coefficient=0.1, size=2):
    corners=[]
    threshold=coefficient*harris_response.max()
    for y in range(size,harris_response.shape[0]-size):
        for x in range(size,harris_response.shape[1]-size):
            if harris_response[y, x]>threshold:
                neighborhood=harris_response[y-size:y+size+1,x-size:x+size+1]
                if harris_response[y,x]==neighborhood.max():
                    corners.append((x,y))
    return corners

image = cv2.imread('D:\\picture\\sudoku.png')
gradient_x, gradient_y=compute_gradients(image)
harris_response=compute_harris_response(gradient_x, gradient_y)
corners=detect_corners(harris_response)
image_with_corners=image.copy()
for corner in corners:
    cv2.circle(image_with_corners,corner,3,(0, 0, 255),-1)
cv2.imwrite('D:\\picture\\result\\sudoku_keypoints.png',image_with_corners)
cv2.imshow('Corners Detected',image_with_corners)
cv2.waitKey(0)
cv2.destroyAllWindows()

image1 = cv2.imread("D:\\picture\\uttower1.jpg")

gradient_x1, gradient_y1=compute_gradients(image1)
harris_response1=compute_harris_response(gradient_x1, gradient_y1)
corners1=detect_corners(harris_response1)
image_with_corners1=image1.copy()
for corner in corners1:
    cv2.circle(image_with_corners1,corner,3,(0, 0, 255),-1)
cv2.imwrite("D:\\picture\\result\\uttower1_keypoints.jpg", image_with_corners1)
image2 = cv2.imread("D:\\picture\\uttower2.jpg")
gradient_x2, gradient_y2=compute_gradients(image2)
harris_response2=compute_harris_response(gradient_x2, gradient_y2)
corners2=detect_corners(harris_response2)
image_with_corners2=image2.copy()
for corner in corners2:
    cv2.circle(image_with_corners2,corner,3,(0, 0, 255),-1)
cv2.imwrite("D:\\picture\\result\\uttower2_keypoints.jpg", image_with_corners2)

