import cv2
import pytesseract
import matplotlib.pyplot as plt 
import numpy as np


pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

og = cv2.imread("samplepaper.jpg")
img = cv2.resize(og, (768, 1024))


    # convert the warped image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # sharpen image
sharpen = cv2.GaussianBlur(gray, (0,0), 3)
sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

        # apply adaptive threshold to get black and white effect
thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)

cv2.imshow("dilation", thresh)
cv2.waitKey(0)

corners = cv2.goodFeaturesToTrack(thresh, maxCorners=100, qualityLevel=0.01, minDistance=10) 
corners = np.int0(corners)
for corner in corners: 
    x, y = corner.ravel() 
    cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
cv2.imshow('Corners', img) 
cv2.waitKey(0)