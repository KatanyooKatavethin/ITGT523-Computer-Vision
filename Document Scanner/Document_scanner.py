import cv2 as cv
import argparse
import numpy as np
import numpy.typing as npt
import pytesseract

def order_points(pts: npt.NDArray) -> npt.NDArray:
    """Order points in clockwise order (top-left, top-right, bottom-right, bottom-left)."""
    rect = np.zeros((4,2), dtype=np.float32)

    #Compute the sum of the (x, y) coordinates for each point
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)] # Top-left
    rect[2] = pts[np.argmax(s)] # Bottom-right

    #Compute the difference between the (x, y) coordinates for each point
    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)] # Top-right
    rect[3] = pts[np.argmax(diff)] # Bottom-left
    return rect

def four_point_transform(image: npt.NDArray, pts: npt.NDArray) -> npt.NDArray:
    """Apply perspective transform to obtain a top-down view Workshop: Building a Document Scanner with Python and OpenCV 5 of an image."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width and height
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl [1]) ** 2)) # Bottom edge
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl [1]) ** 2)) # Top edge
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br [1]) ** 2)) # Right edge
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl [1]) ** 2)) # Left edge
    maxHeight = max(int(heightA), int(heightB))

    # Construct destination points
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1,maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)

    # Compute the Transformation Matrix 
    M = cv.getPerspectiveTransform(rect, dst)

    # Apply perspective transform
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped



# def main() -> None:
#Image path
parser = argparse.ArgumentParser(description= 'Scan a document from the image')
parser.add_argument("-i", '--image', required=True, help="Path to the image to be scanned")
args = parser.parse_args()

#Load image
image = cv.imread(args.image)
if image is None:
        raise ValueError("No image or could not read the image")

#Resize the image
ratio = image.shape[0] / 500.0  
original_image = image.copy()
resized_image = cv.resize(image, (0, 0), fx=500/image.shape[0], fy=500/image.shape[0])

#Convert to grayscale and find document's edges
gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 0)
edged = cv.Canny(blur, 75, 200)

#Contours
contours, _ = cv.findContours(edged, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]

#Find the document contour
screen_cnt = None
for c in contours:
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        screen_cnt = approx
        break


cv.drawContours(resized_image, [screen_cnt], -1, (0, 255, 0), 2)
warped = four_point_transform(original_image, screen_cnt.reshape(4, 2) * ratio)


# Convert to grayscale and apply adaptive thresholding
warped_gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
thresh = cv.adaptiveThreshold(warped_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 10)
text = pytesseract.image_to_string(thresh, lang='eng') # English
print("---------------------------------------------------------")
print(text)
print("---------------------------------------------------------")
#Show final result images
# cv.imshow("contours", contours)
cv.imshow("Original", cv.resize(original_image, (0, 0), fx=650/original_image.shape[0], fy=650/original_image.shape[0]))
cv.imshow("Scanned", cv.resize(thresh, (0, 0), fx=650/thresh.shape[0], fy=650/thresh.shape[0]))
cv.waitKey(0)
cv.destroyAllWindows()