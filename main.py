import cv2
import numpy as np
import glob
import imutils
from stitcher import stitching
image_paths = glob.glob('images/*.jpg')
images = []

for image_path in image_paths:
    image = cv2.imread(image_path)
    print(image_path)
    images.append(image)

# stitch the images together to create a panorama
result = stitching(images)
# show the images
cv2.imshow("Image A", image[0])
cv2.imshow("Image B", image[1])
cv2.imshow("Result", result)
cv2.waitKey(0)

