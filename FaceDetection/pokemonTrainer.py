# Load an image: imread
# Scale an image: cv.resize
# Crop an image: cv.getRectSubPix
# Rotate an image: cv.getRotationMatrix2d, cv.warpAffine
# Replace some part of the image: img1[0:300, 0:300] = img2
# -----------
# Dectect a human face
# Located a human face

import cv2 as cv 

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

input_img = cv.imread('Warf.png')

card_img = cv.imread('bg.png')
card_status = None
gray = cv.cvtColor(input_img,cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.1, 4)

if len(faces) > 0:
    card_status = True
 
    biggest = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = biggest
    cropped = input_img[y:y+h, x:x+w]

    resized = cv.resize(cropped,(0,0), fx=240/cropped.shape[0], fy=240/cropped.shape[0])

    x_offset = -320
    y_offset = 230
    x_end = x_offset + resized.shape[1]
    y_end = y_offset + resized.shape[0]

    card_img[y_offset:y_end, x_offset:x_end] = resized
else:
    card_status = False 
    print("Human face not found")


if card_status == True:
    cv.imshow('Pokemon trainer card', card_img)

cv.waitKey(0)
cv.destroyAllWindows()
