import numpy as np
import cv2
import matplotlib.pyplot as plt

def defect_detect(image):
    img = image.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow("good_Window", hsv)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    blr = cv2.blur(v, (15, 15))
    # cv2.imshow("good_Window1", blr)
    dst = cv2.fastNlMeansDenoising(blr, None, 10, 7, 21)
    _, binary = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_binary = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(adaptive_binary, kernel, iterations=1)
    dilation = cv2.dilate(adaptive_binary, kernel, iterations=1)

    if (dilation == 255).sum() > 1:  # Check if there are any white pixels in dilation
        print("Defective fabric")
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Adjust the area threshold as per your requirement
                cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)
        cv2.imshow("Binary image", adaptive_binary)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Good fabric")

    return adaptive_binary
input_image=cv2.imread('fifth.jpeg')
binary=defect_detect(input_image)











