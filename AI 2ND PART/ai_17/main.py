import cv2
import numpy as np
import imutils
import easyocr

img = cv2.imread("assets/7.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(bfilter, 30, 200)
# getting coordinate
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
#

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
# mask = np.zeros(gray.shape, np.uint8)
# Wrap 'location' in a list to match the expected input for cv2.drawContours
# new_image = cv2.drawContours(mask, [location], 0, 255, -1)
# new_image = cv2.bitwise_and(img, img, mask=mask)
# if location is not None:
#     mask = np.zeros(gray.shape, np.uint8)
#     # Wrap 'location' in a list to match the expected input for cv2.drawContours
#     new_image = cv2.drawContours(mask, [location], 0, 255, -1)
#     new_image = cv2.bitwise_and(img, img, mask=mask)
# else:
#     print("No suitable contour found.")

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1 : x2 + 1, y1 : y2 + 1]

reader = easyocr.Reader(["en"])
result = reader.readtext(cropped_image)
print(result)
# Tambahkan code di bawah ini
text = result[0][-2]
res = cv2.putText(
    img,
    text,
    (approx[0][0][0], approx[1][0][1] + 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2,
)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
cv2.imshow("Image", gray)
cv2.waitKey(0)
cv2.imshow("Image", cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.imshow("Image", new_image)
cv2.waitKey(0)