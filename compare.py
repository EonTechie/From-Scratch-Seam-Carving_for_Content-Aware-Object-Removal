import cv2

# Load the original image
original_image_path = '.\input\chick.jpg'
original_image = cv2.imread(original_image_path)
if original_image is None:
    print("The original image could not be loaded. Please check the file path.")  # Error if the image is not found
    exit()

# Load the mask
mask_path = '.\input\chick_mask.jpg'
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    print("The mask image could not be loaded. Please check the file path.")  # Error if the mask is not found
    exit()

print("mask size: " , mask.shape)  # Print the dimensions of the mask
print("original image size: ",original_image.shape)  # Print the dimensions of the mask