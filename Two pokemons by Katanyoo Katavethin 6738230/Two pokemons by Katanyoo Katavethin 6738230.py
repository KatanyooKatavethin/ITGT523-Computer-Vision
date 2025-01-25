import cv2 as cv
print (cv.__version__)
import numpy as np

#Fetch the images
image = cv.imread("Sharkpedo.png")
image2 = cv.imread("Wooper.png")
image3 = cv.imread("Sharkpedo.png")
image4 = cv.imread("Wooper.png")

#Gray scale the left-side images
gray_image1 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray_image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

#Gray scale the right-side images
gray_image3 = cv.cvtColor(image3, cv.COLOR_BGR2GRAY)
gray_image4 = cv.cvtColor(image4, cv.COLOR_BGR2GRAY)

#Resize the images
resized_img1 = cv.resize(gray_image1, (280, 280))
resized_img2 = cv.resize(gray_image2, (280, 280))

resized_img3 = cv.resize(gray_image3, (280, 280))
resized_img4 = cv.resize(gray_image4, (280, 280))


#Binarizing the images
ret, binary_image = cv.threshold(resized_img3, 90, 255, cv.THRESH_BINARY)
ret, binary_image2 = cv.threshold(resized_img4, 90, 255, cv.THRESH_BINARY)


#Draw a circle
circle_center = (140, 135)
radius = 134
color = (0, 0, 255)
line_thickness = 3
resized_img1 = cv.circle(resized_img1, circle_center, radius, color, line_thickness)


#Make multiple images appear in the same window
left_side = np.concatenate ((resized_img1, resized_img2), axis=0)
right_side = np.concatenate ((binary_image, binary_image2), axis=0)
l_andr = np.concatenate ((left_side, right_side), axis=1)


#Display the images
cv.imshow("The 2 Pokemons!", l_andr)


# cv.imshow("pokemon", gray_image)
cv.waitKey(0)