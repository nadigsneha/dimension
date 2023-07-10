"""
Filename: init.py
Usage: This script will measure different objects in the frame using a reference object of known dimension. 
The object with known dimension must be the leftmost object.
"""
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import imutils
import numpy as np
import cv2




# Function to show array of images (intermediate results)
def show_images(images):
	for i, img in enumerate(images):
		cv2.imshow("image_" + str(i), img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
def crop_image(image, crop_size_cm):
    dpi = 96
    crop_size_px = int(crop_size_cm * dpi / 2.54)
    height, width = image.shape[:2]
    top = crop_size_px
    bottom = height - crop_size_px
    left = crop_size_px
    right = width - crop_size_px
    cropped_image = image[top:bottom, left:right]
    return cropped_image

image = cv2.imread('third.jpeg')
image1=np.copy(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)
edged = cv2.Canny(blur, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
(cnts, _) = contours.sort_contours(cnts)
cnts = [x for x in cnts if cv2.contourArea(x) > 100]
cv2.drawContours(image, cnts, -1, (0,255,0), 3)
ref_object = cnts[0]
box = cv2.minAreaRect(ref_object)
box = cv2.boxPoints(box)
box = np.array(box, dtype="int")
box = perspective.order_points(box)
(tl, tr, br, bl) = box
dist_in_pixel = euclidean(tl, tr)
dist_in_cm = 2
pixel_per_cm = dist_in_pixel/dist_in_cm
for cnt in cnts:
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect
    object_width = w / pixel_per_cm
    object_height = h / pixel_per_cm
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.polylines(image, [box], True, (255, 0, 0), 2)

    print(object_width, object_height)
    cv2.putText(image, "{:.1f}cm".format(object_width), (int(x-10), int(y-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(image, "{:.1f}cm".format(object_height), (int(x-10), int(y+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Resized_Window", 450, 600)
cv2.imshow("Resized_Window", image)
show_images([image])
cropped_image = crop_image(image1, 6)
output_path = 'cropped_image1.jpg'
cv2.imwrite(output_path, cropped_image)

# fig,ax=plt.subplots(2,3,figsize=(15,10))
# ax[0,0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
# ax[0,0].set_title('Original')
#
# ax[0,1].imshow(cv2.cvtColor(hsv,cv2.COLOR_BGR2RGB))
# ax[0,1].set_title('HSV')
#
# ax[0,2].imshow(cv2.cvtColor(v,cv2.COLOR_BGR2RGB))
# ax[0,2].set_title('vlaue')
#
# ax[1,0].imshow(cv2.cvtColor(blr,cv2.COLOR_BGR2RGB))
# ax[1,0].set_title('BLUR')
#
# ax[1,1].imshow(cv2.cvtColor(dst,cv2.COLOR_BGR2RGB))
# ax[1,1].set_title('FILTER')
#
# ax[1,2].imshow(dilation,cmap='gray')
# ax[1,2].set_title('BINARY')
#
#
# fig.tight_layout()

