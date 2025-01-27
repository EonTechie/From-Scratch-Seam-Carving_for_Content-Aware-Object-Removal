import cv2
import numpy as np

# Load the image
image_path = 'object_m.jpg'  # Path to the input image
image = cv2.imread(image_path)
if image is None:
    print("Failed to load the image. Please check the file path.")  # Error message if the image is not found
    exit()

# Convert to grayscale to simplify processing
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale

# Create a mask for non-black pixels
_, mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)  # Threshold to set non-black pixels to white

# Display the original image and the mask
cv2.imshow("Original Image", image)  # Show the original image
cv2.imshow("White Mask", mask)  # Show the mask where non-black pixels are white
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the generated mask
cv2.imwrite('object_mask.jpg', mask)  # Save the mask as a file
