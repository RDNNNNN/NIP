# --image license_plate.png --method left-to-right
# --image license_plate.png --method right-to-left

""" Page 6 """
# import the necessary packages
import argparse

import cv2
import imutils
import numpy as np


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(
        *sorted(
            zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse
        )
    )

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


""" Page 7 """
def draw_contour(image, c, i):
    # compute the center of the contour area and draw a circle
    # representing the center
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # draw the contour number on the image
    # cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),2)
    cv2.putText(
        image,
        f"#{i + 1}",
        (cX - 20, cY - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        1,
    )

    # return the image with the contour number drawn on it
    return image


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-m", "--method", required=True, help="Sorting method")
args = vars(ap.parse_args())


# load the image and initialize the accumulated edge image
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
# print("Otsu's thresholding value: {}".format(T))
print(f"Otsu's thresholding value: {T}")

i = 0
eroded = cv2.erode(threshInv.copy(), None, iterations=i + 1)
# cv2.imshow("Eroded {} times".format(i + 1), eroded)
cv2.imshow(f"Eroded {i + 1} times", eroded)

dilated = cv2.dilate(eroded.copy(), None, iterations=i + 1)
# cv2.imshow("Dilated {} times".format(i + 1), dilated)
cv2.imshow(f"Dilated {i + 1} times", dilated)


""" Page 8 """
# find contours in the accumulated image, keeping only the largest ones
cnts = cv2.findContours(
    dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[1:8]
orig = image.copy()

# loop over the (unsorted) contours and draw them
for i, c in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 1)
    orig = draw_contour(orig, c, i)

# show the original, unsorted contour image
cv2.imshow("Unsorted", orig)
orig = image.copy()

# sort the contours according to the provided method
(cnts, boundingBoxes) = sort_contours(cnts, method=args["method"])

# loop over the (now sorted) contours and draw them
for i, c in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 1)
    draw_contour(orig, c, i)

# show the output image
cv2.imshow("Sorted", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
