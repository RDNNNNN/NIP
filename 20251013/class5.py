# --image ut.jpg

""" Page 10 """
# import the necessary packages
import argparse

import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# Load the image and grab each channel: Red, Green, and Blue. It's
# important to note that OpenCV stores an image as NumPy array with
# its channels in reverse order! When we call cv2.split, we are
# actually getting the channels as Blue, Green, Red!
image = cv2.imread(args["image"])
(B, G, R) = cv2.split(image)

# show each channel individually
cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)
cv2.waitKey(0)

# merge the image back together again
merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# visualize each channel in color
zeros = np.zeros(image.shape[:2], dtype="uint8")
cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))
cv2.waitKey(0)

white = np.ones(image.shape[:2], dtype="uint8") * 255
cv2.imshow("Red", cv2.merge([white, white, R]))
cv2.imshow("Green", cv2.merge([white, G, white]))
cv2.imshow("Blue", cv2.merge([B, white, white]))
cv2.waitKey(0)

white = np.zeros(image.shape[:2], dtype="uint8") * 255
cv2.imshow("Red", cv2.merge([white, white, zeros]))
cv2.imshow("Green", cv2.merge([white, zeros, white]))
cv2.imshow("Blue", cv2.merge([white, white, white]))
cv2.waitKey(0)
