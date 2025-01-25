import cv2 as cv
import numpy as np
img = cv.imread("  .png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blurred = cv.GaussianBulr(gray, (5,5), 0)

sobelx = cv.Sobel(blurred, cv.CV_64F, 1,0, ksise=3)
sobely = cv.Sobel(blurred, cv.CV_64F, 0,1, ksise=3)
sobel_combined = cv.magitude(sobelx, sobely)
sobel_combined = sobel_combined.astype(np.uint8)
_, threshold_img = cv.threshold(sobel_combined, 100, 255, cv.THRESH_BINARY_INV)

anime = cv.bitwise_and(img, img, mask=threshold_img)

#cv.imshow("blurr", blurred)
#cv.imshow("sobel x", sobelx)
#cv.imshow("sobel y", sobely)
cv.imshow("sobel", sobel_combined)
cv.imshow("anime", anime)
cv.imshow("edge", threshold_img)

cv.waitKey(0)
cv.destroyAllWindows()