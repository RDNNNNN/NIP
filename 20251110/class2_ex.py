# --image license_plate.png

# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# show the original image
cv2.imshow("Original", image)
cv2.imshow("blurred", blurred)

(T, threshInv) = cv2.threshold(
    blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
)
cv2.imshow("Threshold", threshInv)
print("Otsu's thresholding value: {}".format(T))

i = 0
eroded = cv2.erode(threshInv.copy(), None, iterations=i + 1)
cv2.imshow("Eroded {} times".format(i + 1), eroded)

dilated = cv2.dilate(eroded.copy(), None, iterations=i + 1)
cv2.imshow("Dilated {} times".format(i + 1), dilated)

# find all contours in the image and draw ALL contours on the image
cnts = cv2.findContours(eroded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
clone = image.copy()
cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
print("Found {} contours".format(len(cnts)))

# show the output image
cv2.imshow("All Contours", clone)
cv2.waitKey(0)

# re-clone the image and close all open windows
clone = image.copy()
cv2.destroyAllWindows()

# loop over the contours individually and draw each of them
for i, c in enumerate(cnts):
    print("Drawing contour #{}".format(i + 1))
    cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)
    cv2.imshow("Single Contour", clone)
    cv2.waitKey(0)

# re-clone the image and close all open windows
clone = image.copy()
cv2.destroyAllWindows()

# find contours in the image, but this time keep only the EXTERNAL
# contours in the image
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
print("Found {} EXTERNAL contours".format(len(cnts)))

# show the output image
cv2.imshow("All Contours", clone)
cv2.waitKey(0)

# re-clone the image and close all open windows
clone = image.copy()
cv2.destroyAllWindows()

# loop over the contours individually
for c in cnts:
    # construct a mask by drawing only the current contour
    mask = np.zeros(gray.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

# show the images
cv2.imshow("Image", image)
cv2.imshow("Mask", mask)
cv2.imshow("Image + Mask", cv2.bitwise_and(image, image, mask=mask))
cv2.waitKey(0)
