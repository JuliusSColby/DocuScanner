import cv2

# Load the image
image_path = "cropped_final.jpg"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image
_, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

x_min, y_min, x_max, y_max = float("inf"), float("inf"), 0, 0

# Iterate through contours to find the overall bounding box
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    x_min = min(x_min, x)
    y_min = min(y_min, y)
    x_max = max(x_max, x + w)
    y_max = max(y_max, y + h)

# Add padding around the bounding box
padding = 10  # Adjust this value to control box size
x_min = max(x_min - padding, 0)
y_min = max(y_min - padding, 0)
x_max = min(x_max + padding, image.shape[1])
y_max = min(y_max + padding, image.shape[0])

# Draw the bounding box
cv2.rectangle(gray, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Show the image with the box
cv2.imshow("Bounding Box", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite("larger_bounding_box.png", gray)




# Draw bounding boxes around contours
#for contour in contours:
#    x, y, w, h = cv2.boundingRect(contour)
#   cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the result
cv2.imshow("thresh", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()