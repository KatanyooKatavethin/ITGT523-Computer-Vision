import cv2 as cv

color_offset = 0
def nothing(x):
    pass

title_window = "The Hulk"
cv.namedWindow(title_window)
cv.createTrackbar("color_offset", title_window, 0, 255, nothing)


image = cv.imread("Pokemon card.png")
#Resize the images
resized_img = cv.resize(image, (286, 400))
cv.imshow("Pokemon", resized_img)

while True:

      img_hsv = cv.cvtColor(resized_img, cv.COLOR_BGR2HSV)

      color_offset = cv.getTrackbarPos("color_offset", title_window)

      #hsv image processing...
      #print(img_hsv [0] [0] [0])

      #first : y axis or the row
      #second : x axis or the column
      #third : channels (HSV)
      #offset = 100
      for row in range(0, img_hsv.shape[0]):
          for column in range(0, img_hsv.shape[1]):
              img_hsv [row] [column] [0] = img_hsv [row] [column] [0] + color_offset


      img_final = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)

      #cv.imshow("Pokemon hsv", img_hsv)
      cv.imshow(title_window, img_final)

      if cv.waitKey(1) & 0xFF == 27:
           break

cv.destroyAllWindows()