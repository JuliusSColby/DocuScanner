import cv2
import numpy as np

# Load the image
og = cv2.imread("samplepaper.jpg")
image = cv2.resize(og, (768, 1024))

# Convert the image to HSV color space

#figure out how to actually make a color range that works




gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=500)
corners = np.int0(corners)

# Draw circles around detected corners
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

#cv2.imshow('gray', gray )
cv2.imshow('corners', image)
cv2.waitKey(0)


"""
# Define the lower and upper bounds for gray color
lower_gray = np.array([0, 0, 50])
upper_gray = np.array([180, 50, 255])

# Create a binary mask where gray is within the range
mask = cv2.inRange(hsv, lower_gray, upper_gray)

# Apply the mask to get the gray regions
result = cv2.bitwise_and(image, image, mask=mask)

# Convert the result to grayscale for corner detection
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# Detect corners using Shi-Tomasi method
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
corners = np.int0(corners)

# Draw circles around detected corners
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

# Display the result
cv2.imshow('Gray Regions and Corners', image)
cv2.imshow('Mask', mask)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""