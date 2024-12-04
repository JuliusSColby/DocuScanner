import cv2
import pytesseract
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

og = cv2.imread("test2.jpg")
img = cv2.resize(og, (768, 1024))


    # convert the warped image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # sharpen image
blurred = cv2.GaussianBlur(gray, (0,0), 3)
sharpen = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

        # apply adaptive threshold to get black and white effect
thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)

#cv2.imshow("thresh", thresh)
#cv2.waitKey(0)


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

# Get the bounding box of the document contour 
x, y, w, h = cv2.boundingRect(document_contour) 
# Crop the image to the bounding box 
cropped_image = img[y:y+h, x:x+w] 
# Save the cropped image 
cv2.imwrite("cropped_document.jpg", cropped_image) 


#PART 2: TEXT EXTRACTION

#CROPPING
image_path = "cropped_document.jpg"
image = Image.open(image_path)

# Get image dimensions
width, height = image.size

# Calculate the crop margins (5% of width and height)
crop_margin_width = int(width * 0.05)
crop_margin_height = int(height * 0.05)

# Define the crop box (left, upper, right, lower)
crop_box = (
    crop_margin_width,                          # Left
    crop_margin_height,                         # Top
    width - crop_margin_width,                  # Right
    height - crop_margin_height                 # Bottom
)

# Crop the image
cropped_final = image.crop(crop_box)
cropped_final.save("cropped_final.jpg")

cf = cv2.imread("cropped_final.jpg")
# Convert the cropped image to grayscale for text extraction
cropped_gray = cv2.cvtColor(cf, cv2.COLOR_BGR2GRAY)
blurred2 = cv2.GaussianBlur(cropped_gray, (0,0), 3)
sharpen2 = cv2.addWeighted(cropped_gray, 1.5, blurred2, -0.5, 0)

        # apply adaptive threshold to get black and white effect
thresh2 = cv2.adaptiveThreshold(sharpen2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)

contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, 
                                                 cv2.CHAIN_APPROX_NONE)




#corners = cv2.goodFeaturesToTrack(thresh2, maxCorners=50, qualityLevel=0.02, minDistance=10) 
#corners = np.int0(corners)
#for corner in corners: 
#    x, y = corner.ravel() 
#    cv2.circle(thresh2, (x, y), 3, (0, 255, 0), -1)
#cv2.imshow('Corners', thresh2) 
#cv2.waitKey(0)

# Extract text from the cropped image using pytesseract
#extracted_text = pytesseract.image_to_string(thresh2, lang='eng')

# Print the extracted text
#print("Extracted Text:")
#print(extracted_text)

for cnt in contours2:
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Drawing a rectangle on copied image
    rect = cv2.rectangle(thresh2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #SHOWIMAGE
    
    # Cropping the text block for giving input to OCR
    cropped = thresh2[y:y + h, x:x + w]
    
    # Open the file in append mode
    #file = open("recognized.txt", "a")
    
    # Apply OCR on the cropped image
    text = pytesseract.image_to_string(cropped)
print(text)
cv2.imshow("thresh2", thresh2)
cv2.waitKey(0)
    # Appending the text into file
    #file.write(text)
    #file.write("\n")
    
    # Close the file
    #file.close()












"""
cv2.imshow("Cropped Document", cropped_image) 
cv2.imshow("Document Border", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""