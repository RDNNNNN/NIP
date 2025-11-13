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
    # compute the area and the perimeter of the contour
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    print(
        "Contour #{} -- area: {:.2f}, perimeter: {:.2f}".format(
            i + 1, area, perimeter
        )
    )

    if area > 880 and area < 2500:
        # draw the contour on the image
        # cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)
        #
        # # compute the center of the contour and draw the contour number
        # M = cv2.moments(c)
        # cX = int(M["m10"] / M["m00"])
        # cY = int(M["m01"] / M["m00"])
        # cv2.putText(clone, "#{}".format(i + 1), (cX - 3, cY), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5, (255, 255, 255), 1)
        # Rectangle
        # fit a bounding box to the contour
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 1)

        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # show the images
        cv2.imshow("Image", image)
        cv2.imshow("Mask", mask)
        cv2.imshow("Image + Mask", cv2.bitwise_and(image, image, mask=mask))

        # show the output image
        cv2.imshow("Contours", clone)
        cv2.waitKey(0)

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
