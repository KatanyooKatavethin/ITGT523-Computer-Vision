import cv2 as cv
import numpy as np
img = cv.imread("Coin and pills.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blurred = cv.GaussianBlur(gray, (9,9), 0)

sobelx = cv.Sobel(blurred, cv.CV_64F, 1,0, ksize=3)
sobely = cv.Sobel(blurred, cv.CV_64F, 0,1, ksize=3)
sobel_combined = cv.magnitude(sobelx, sobely)
sobel_combined = sobel_combined.astype(np.uint8)
_, threshold_img = cv.threshold(sobel_combined, 100, 255, cv.THRESH_BINARY_INV)

anime = cv.bitwise_and(img, img, mask=threshold_img)
canny = cv.Canny(blurred, threshold1=90, threshold2=110)

_, threshold_img2 = cv.threshold(canny, 50, 155, cv.THRESH_BINARY)

contours, _ = cv.findContours(threshold_img2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

print("Number of objects in the picture:", len(contours))

result = img.copy()
cv.drawContours(result, contours, -1, (0, 255, 0), 2)

#cv.imshow("blur", blurred)


#cv.imshow("sobel x", sobelx)
#cv.imshow("sobel y", sobely)
# cv.imshow("sobel", sobel_combined)
# cv.imshow("anime", anime)
# cv.imshow("edge", threshold_img2)
# cv.imshow("canny", canny)
cv.imshow("contours", result)

cv.waitKey(0)
cv.destroyAllWindows()