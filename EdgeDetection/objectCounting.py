import cv2 as cv
import numpy as np

#First picture
img_1 = cv.imread("example_01.png")

gray_1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
blurred_1 = cv.GaussianBlur(gray_1, (11,11), 0)

canny_1 = cv.Canny(blurred_1, threshold1=50, threshold2=150)
kernel = np.ones((5,5), np.uint8)
dilation = cv.dilate(canny_1, kernel, iterations=1)

contours_1, _ = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
min_Area = 150
contour_Area = [c for c in contours_1 if cv.contourArea(c)>= min_Area]
result = img_1.copy()
cv.drawContours(result, contour_Area, -1, (0, 255, 0), 2)

print("Number of objects in picture 1:", len(contour_Area))


#Second picture
img_2 = cv.imread("example_02.png")

gray_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
blurred_2 = cv.GaussianBlur(gray_2, (11,11), 0)

canny_2 = cv.Canny(blurred_2, threshold1=50, threshold2=150)
dilation_2 = cv.dilate(canny_2, kernel, iterations=1)

contours_2, _ = cv.findContours(dilation_2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contour_Area_2 = [c for c in contours_2 if cv.contourArea(c)>= min_Area]
result_2 = img_2.copy()
cv.drawContours(result_2, contour_Area_2, -1, (0, 255, 0), 2)

print("Number of objects in picture 2:",len(contour_Area_2))


#Third picture
img_3 = cv.imread("example_03.png")

gray_3 = cv.cvtColor(img_3, cv.COLOR_BGR2GRAY)
blurred_3 = cv.GaussianBlur(gray_3, (11,11), 0)

canny_3 = cv.Canny(blurred_3, threshold1=50, threshold2=150)
dilation_3 = cv.dilate(canny_3, kernel, iterations=1)

contours_3, _ = cv.findContours(dilation_3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contour_Area_3 = [c for c in contours_3 if cv.contourArea(c)>= min_Area]
result_3 = img_3.copy()
cv.drawContours(result_3, contour_Area_3, -1, (0, 255, 0), 2)

print("Number of objects in picture 3:",len(contour_Area_3))


cv.imshow("Picture 1", result)
cv.imshow("Picture 2", result_2)
cv.imshow("Picture 3", result_3)

cv.waitKey(0)
cv.destroyAllWindows()