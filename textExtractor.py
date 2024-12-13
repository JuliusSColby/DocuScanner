import cv2
import pytesseract
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

#either input file name (without extension) via prompt or hard code it
#img_name = "test6"
img_name = input("Enter path/name of file: ")

og = cv2.imread(img_name + ".jpg")

img = cv2.resize(og, (768, 1024))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (0,0), 3)
sharpen = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
blurred2 = cv2.GaussianBlur(thresh, (5, 5), 0)
edges = cv2.Canny(blurred2, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
document_contour = contours[0]
epsilon = 0.02 * cv2.arcLength(document_contour, True)
approx = cv2.approxPolyDP(document_contour, epsilon, True)
output = img.copy()
cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)

x, y, w, h = cv2.boundingRect(document_contour) 
cropped_image = img[y:y+h, x:x+w] 
cv2.imwrite(img_name + "_cropped.jpg", cropped_image) 

#PART 2: TEXT EXTRACTION

#CROPPING
image_path = img_name + "_cropped.jpg"
image = Image.open(image_path)

width, height = image.size

crop_margin_width = int(width * 0.10)
crop_margin_height = int(height * 0.10)
crop_box = (crop_margin_width, crop_margin_height, width - crop_margin_width, height - crop_margin_height)
cropped_final = image.crop(crop_box)
cropped_final.save(image_path)

cf = cv2.imread(image_path)
gray2 = cv2.cvtColor(cf, cv2.COLOR_BGR2GRAY)
_, thresh2 = cv2.threshold(gray2, 75, 255, cv2.THRESH_BINARY_INV)
contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

x_min, y_min, x_max, y_max = float("inf"), float("inf"), 0, 0
for contour in contours2:
    x, y, w, h = cv2.boundingRect(contour)
    x_min = min(x_min, x)
    y_min = min(y_min, y)
    x_max = max(x_max, x + w)
    y_max = max(y_max, y + h)
padding = 10
x_min = max(x_min - padding, 0)
y_min = max(y_min - padding, 0)
x_max = min(x_max + padding, cf.shape[1])
y_max = min(y_max + padding, cf.shape[0])
cv2.rectangle(gray2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
cropped = gray2[y_min:y_max, x_min:x_max]
    
file = open(img_name + "_recognized.txt", "w")

scale_factor = 1
enhanced = cv2.resize(cropped, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
text = pytesseract.image_to_string(enhanced)

print(text)
file.write(text)
file.write("\n")
file.close()

# Show the image with the box
cv2.imshow("cropped", enhanced)
cv2.imshow("Bounding Box", gray2)
cv2.waitKey(0)
cv2.destroyAllWindows()



