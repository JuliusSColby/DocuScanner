import cv2
import pytesseract
import matplotlib.pyplot as plt 
import numpy as np

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'  # In case using colab after installing above modules

# Read image from which text needs to be extracted
#img = cv2.imread("sample.jpg")
og = cv2.imread("samplepaper.jpg")
img = cv2.resize(og, (768, 1024))
# Preprocessing the image starts

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, threshold1=50, threshold2=150)

#SHOWIMAGE
cv2.imshow("gray", edges)
cv2.waitKey(0)
#plt.imshow(thresh1)

#alt (finding corners)
corners = cv2.goodFeaturesToTrack(gray, maxCorners=4, qualityLevel=0.01, minDistance=100) 
corners = np.int0(corners)
for corner in corners: 
    x, y = corner.ravel() 
    cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
cv2.imshow('Corners', img) 
cv2.waitKey(0)



# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Specify structure shape and kernel size. 
# Kernel size increases or decreases the area 
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect 
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

#SHOWIMAGE
cv2.imshow("dilation", dilation)
cv2.waitKey(0)


# Finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
                                                 cv2.CHAIN_APPROX_NONE)


# Creating a copy of image
im2 = img.copy()

# A text file is created and flushed
file = open("recognized.txt", "w+")
file.write("")
file.close()

# Looping through the identified contours
# Then rectangular part is cropped and passed on
# to pytesseract for extracting text from it
# Extracted text is then written into the text file
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Drawing a rectangle on copied image
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #SHOWIMAGE
    cv2.imshow("gray", im2)
    cv2.waitKey(0)
    
    # Cropping the text block for giving input to OCR
    cropped = im2[y:y + h, x:x + w]
    
    # Open the file in append mode
    file = open("recognized.txt", "a")
    
    # Apply OCR on the cropped image
    text = pytesseract.image_to_string(cropped)
    
    # Appending the text into file
    file.write(text)
    file.write("\n")
    
    # Close the file
    file.close()

# This code is modified by Susobhan Akhuli