import cv2
import numpy as np
import glob
import imutils
from stitcher import stitcher
image_paths = glob.glob('images/*.jpg')
# image_paths = glob.glob('images/*.png')
images = []

for image_path in image_paths:
    image = cv2.imread(image_path)
    print(image_path)
    images.append(image)

print(len(images))
# stitch the images together to create a panorama
stitcher(images)
