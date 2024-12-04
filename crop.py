from PIL import Image
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
cropped_image = image.crop(crop_box)

# Save and display the cropped image
cropped_image.save("cropped_image_5_percent.jpg")
cropped_image.show()