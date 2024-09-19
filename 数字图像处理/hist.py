import cv2
import numpy as np
import matplotlib.pyplot as plt

def restore(underwater_image):

    # 将图像转换到HSV颜色空间
    hsv_image = cv2.cvtColor(underwater_image, cv2.COLOR_BGR2HSV)

    # 对亮度通道进行直方图均衡化
    hsv_image[:,:,2] = cv2.equalizeHist(hsv_image[:,:,2])

    # 将图像转换回原始颜色空间
    enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return enhanced_image

if __name__ == '__main__':
    image_path_='other\\Project\\water\\'
    image_path_num = [3,16,47,246,554,2129,2552,5015]
    for num in image_path_num :
        image_path = image_path_ + str(num) + '.jpg'
        img=cv2.imread(image_path)
        img=cv2.resize(img,(600,450))
        restored_img = restore(img)
        cv2.imshow('result',np.hstack((img,restored_img))) # type: ignore
        cv2.waitKey()
        cv2.destroyAllWindows()




