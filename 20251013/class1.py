""" Page 5 """
# import the necessary packages
import cv2

# load the image and show it
image = cv2.imread("ut.jpg")
cv2.imshow("Original", image)

# cropping an image is accomplished using simple NumPy array slices --
# let's crop the face from the image
face = image[65:403, 98:393]
cv2.imshow("Face", face)
cv2.waitKey(0)

# ...and now let's crop the entire body
body = image[264:326, 409:1006]
cv2.imshow("Body", body)
cv2.waitKey(0)