""" Page 26 """
# import the necessary packages
import cv2
import imutils

# load the tic-tac-toe image and convert it to grayscale
image = cv2.imread("tictactoe.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# find all contours on the tic-tac-toe board
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

""" Page 27 """
# loop over the contours
for (i, c) in enumerate(cnts):
    # compute the area of the contour along with the bounding box
    # to compute the aspect ratio
    area = cv2.contourArea(c)
    (x, y, w, h) = cv2.boundingRect(c)

    # compute the convex hull of the contour, then use the area of the
    # original contour and the area of the convex hull to compute the
    # solidity
    hull = cv2.convexHull(c)
    hullArea = cv2.contourArea(hull)
    solidity = area / float(hullArea)

    # initialize the character text
    char = "?"

    # if the solidity is high, then we are examining an `O`
    if solidity > 0.9:
        char = "O"
    # otherwise, if the solidity is still reasonably high, we
    # are examining an `X`
    elif solidity > 0.5:
        char = "X"

    # if the character is not unknown, draw it
    if char != "?":
        cv2.drawContours(image, [c], -1, (0, 255, 0), 3)
        cv2.putText(
            image,
            char,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            (0, 255, 0),
            4,
        )

    # show the contour properties
    # print("{} (Contour #{}) -- solidity={:.2f}".format(char, i + 1, solidity))
    print(f"{char} (Contour #{i + 1}) -- solidity={solidity:.2f}")

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
