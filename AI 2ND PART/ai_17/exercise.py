import cv2
import numpy as np
img = cv2.imread('assets/shape.jpg')
imgGrayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGrayScale, (7,7), 1)  
imgCanny = cv2.Canny(imgBlur, 50,50)

def getContours(img):
  contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  for cnt in contours:
    area = cv2.contourArea(cnt)
    print(area)
    cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
imgContour = img.copy()
getContours(imgCanny)
cv2.imshow("Contouring Image", imgContour)
cv2.imshow("Original Picture", img)
cv2.imshow("Gray Scale", imgGrayScale)
cv2.imshow("Blurry Image", imgBlur)
cv2.imshow("Canny Edge Detector", imgCanny)



cv2.waitKey(0)