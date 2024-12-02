import cv2
import numpy as np

# Load the image
img = cv2.imread('blackwhite.png')
img_resized = cv2.resize(img, (768, 1024))

# Convert to grayscale
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur for smoothing
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area (largest first) and keep the largest one
contours = sorted(contours, key=cv2.contourArea, reverse=True)
document_contour = contours[0]

# Approximate the contour to a polygon
epsilon = 0.02 * cv2.arcLength(document_contour, True)
approx = cv2.approxPolyDP(document_contour, epsilon, True)

# Draw the contour on the image
output = img_resized.copy()
cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)

# Show the result
cv2.imshow("Document Border", output)
cv2.waitKey(0)
cv2.destroyAllWindows()