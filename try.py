import cv2
import pytesseract
import matplotlib.pyplot as plt 
import numpy as np


pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

og = cv2.imread("sample2.jpg")
img = cv2.resize(og, (768, 1024))


    # convert the warped image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # sharpen image
blurred = cv2.GaussianBlur(gray, (0,0), 3)
sharpen = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

        # apply adaptive threshold to get black and white effect
thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)

cv2.imshow("thresh", thresh)
cv2.waitKey(0)


#gray2 = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur for smoothing
blurred2 = cv2.GaussianBlur(thresh, (5, 5), 0)

# Edge detection using Canny
edges = cv2.Canny(blurred2, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area (largest first) and keep the largest one
contours = sorted(contours, key=cv2.contourArea, reverse=True)
document_contour = contours[0]

# Approximate the contour to a polygon
epsilon = 0.02 * cv2.arcLength(document_contour, True)
approx = cv2.approxPolyDP(document_contour, epsilon, True)

# Draw the contour on the image
output = img.copy()
cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)

# Show the result
cv2.imshow("Document Border", output)
cv2.waitKey(0)
cv2.destroyAllWindows()






"""
corners = cv2.goodFeaturesToTrack(thresh, maxCorners=100, qualityLevel=0.01, minDistance=10) 
corners = np.int0(corners)
for corner in corners: 
    x, y = corner.ravel() 
    cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
cv2.imshow('Corners', img) 
cv2.waitKey(0)
"""