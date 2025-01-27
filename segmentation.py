import cv2
import numpy as np

def add_image(image_path):
    """
    Loads and returns a color image from the given path, 
    while printing the filename, its dimensions, and channel count.

    Args:
        image_path (str): Path to the input image.

    Returns:
        numpy.ndarray: Loaded image in RGB format.
    """
    image = cv2.imread(image_path)
    if image is not None:
        height, width, channels = image.shape
        print(f"Loaded '{image_path}' with dimensions: {width}x{height}x{channels}")
    else:
        print(f"Failed to load image: {image_path}")
    return image



def add_mask(mask_path):
    """
    Loads and returns a grayscale mask from the given path.

    Args:
        mask_path (str): Path to the mask image.

    Returns:
        numpy.ndarray: Loaded mask in grayscale format.
    """
    return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)



def compute_energy(image):
    """
    Computes the energy of an image using horizontal and vertical gradients.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Energy matrix.
    """
    # Compute gradients using vectorized operations
    horizontal_gradient = np.abs(np.diff(image, axis=1, append=image[:, -1:, :]))
    vertical_gradient = np.abs(np.diff(image, axis=0, append=image[-1:, :, :]))
    
    # Sum gradients across color channels
    energy = np.sum(horizontal_gradient, axis=2) + np.sum(vertical_gradient, axis=2)
    return energy


def reweight_energy(energy, mask):
    """
    Reweights the energy matrix based on a given mask.

    Args:
        energy (np.ndarray): Original energy matrix.
        mask (np.ndarray): Mask matrix (grayscale).

    Returns:
        np.ndarray: Reweighted energy matrix.
    """
    reweighted_energy = energy.copy()  # Make a copy of the energy matrix
    reweighted_energy[mask > 0] = 0  # Set energy to 0 where mask is greater than 0
    return reweighted_energy



def find_vertical_path(reweighted_energy):
    """
    Finds the vertical path with the least energy using the reweighted energy matrix.
    Args:
        reweighted_energy (np.ndarray): The reweighted energy matrix.
    Returns:
        list: The vertical path with the least energy (list of coordinates).
    """
    height, width = reweighted_energy.shape
    cumulative_energy = np.zeros_like(reweighted_energy)
    path = np.zeros_like(reweighted_energy, dtype=int)

    # Transfer the first row's energy values
    cumulative_energy[0, :] = reweighted_energy[0, :]

    # Compute the cumulative energy matrix
    for y in range(1, height): #because we took the first row as it is
        for x in range(width): 
            min_energy = cumulative_energy[y-1, x]  # Top middle
            min_index = x

            # Top left
            if x > 0 and cumulative_energy[y-1, x-1] < min_energy:
                min_energy = cumulative_energy[y-1, x-1]
                min_index = x-1

            # Top right
            if x < width-1 and cumulative_energy[y-1, x+1] < min_energy:
                min_energy = cumulative_energy[y-1, x+1]
                min_index = x+1

            # Update cumulative energy value
            cumulative_energy[y, x] = reweighted_energy[y, x] + min_energy
            path[y, x] = min_index

    # Find the pixel with the least energy in the last row
    min_index = np.argmin(cumulative_energy[-1, :])
    total_energy = cumulative_energy[-1, min_index] # its value is the energy

    # Backtrack to trace the path
    vertical_path = [(height-1, int(min_index))]  # Save the column index as an integer
    for y in range(height-1, 0, -1):
        min_index = path[y, min_index]
        vertical_path.append((y-1, int(min_index)))  # Save as integer here as well

    vertical_path.reverse()
    return vertical_path, total_energy




def remove_vertical_path(image, path):
    """
    Removes the least energy vertical path from an image or mask and resizes it.
    Args:
        image (np.ndarray): The original image or mask (can be 2D or 3D).
        path (list): Coordinates of the least energy path [(y, x), ...].
    Returns:
        np.ndarray: Updated image or mask (with one column removed).
    """
    if len(image.shape) == 3:  # Colored image (3D)
        height, width, channels = image.shape
        new_image = np.zeros((height, width - 1, channels), dtype=image.dtype)
        
        for y, x in path:
            # Remove the pixel at the specified path (x coordinate) for each row
            new_image[y, :, :] = np.delete(image[y, :, :], x, axis=0) # in y row, removes x column

    elif len(image.shape) == 2:  # Grayscale image (2D)
        height, width = image.shape
        new_image = np.zeros((height, width - 1), dtype=image.dtype)
        
        for y, x in path: # tuple unpacking
            # Remove the pixel at the specified path (x coordinate) for each row
            new_image[y, :] = np.delete(image[y, :], x, axis=0)

    return new_image



def iterative_object_removal(image, mask, reweighted_energy, max_iter=50):
    """
    Iteratively removes an object from the image using the mask and reweighted energy matrix.
    Args:
        image (np.ndarray): Original image (can be colored or grayscale).
        mask (np.ndarray): Binary mask defining the object to be removed.
        reweighted_energy (np.ndarray): Initial reweighted energy matrix.
    Returns:
        np.ndarray: Updated image after object removal.
    """
    iteration = 0
    while np.any(reweighted_energy == 0) and np.any(mask > 0) and iteration<max_iter :
        
        
        path, total_energy = find_vertical_path(reweighted_energy)
        

        # Remove the path from the original image
        image = remove_vertical_path(image, path)

        reweighted_energy = remove_vertical_path(reweighted_energy, path)
        
        # Update the mask in the same way
        mask = remove_vertical_path(mask, path)
        
        # Stop if the image width has reached its minimum limit
        if image.shape[1] <= 1:
            print("Image width has reached the minimum limit.")
            break

        iteration += 1
    
    return image


def restore_image_to_original_size(image, original_width):
    """
    Restores the image to its original size by adding back the least energy paths.
    Args:
        image (np.ndarray): Resized image.
        original_width (int): Original width of the image.
    Returns:
        np.ndarray: Image restored to its original size.
    """
    current_width = image.shape[1] # 0: row number, 1: column number

    while current_width < original_width:
        # (a) Recalculate the energy matrix
        energy_matrix = compute_energy(image)

        # (b) Find the vertical path with the least energy
        vertical_path, _ = find_vertical_path(energy_matrix)

        # (c) Extend the image by duplicating the found vertical path
        image = add_vertical_path(image, vertical_path)

        # Update the current width of the image
        current_width = image.shape[1]
        # print(f"Updated Image Shape: {image.shape}")

    return image



def add_vertical_path(image, path):
    """
    Adds the least energy vertical path back to the image by duplicating it.
    Args:
        image (np.ndarray): The current image.
        path (list): Coordinates of the vertical path to be added [(y, x), ...].
    Returns:
        np.ndarray: Updated image with one column added.
    """
    height, width, channels = image.shape
    # Create a new image with one additional column
    new_image = np.zeros((height, width + 1, channels), dtype=image.dtype)

    for y, x in path:
        
        # Copy pixels up to the path column
        new_image[y, :x, :] = image[y, :x, :] # pixels from 0 to x-1 column, pre-path column
        
        # identifies the pixels of y row up to the path column
        
        # Duplicate the selected pixel
        new_image[y, x, :] = image[y, x, :]
        
        # Copy pixels after the path column
        new_image[y, x + 1:, :] = image[y, x:, :]

    return new_image




# Load the images
cat_image = add_image('input/cat.jpg')
sheep_image = add_image("input/sheep.jpg")
dog_image = add_image("input/dog.jpg")



def process_image(image, mask_path, output_name):
    """Processes an image: computes energy, reweights, removes path, and saves the result."""
    energy_matrix = compute_energy(image)
    mask = add_mask(mask_path)
    reweighted_energy = reweight_energy(energy_matrix, mask)
    removed_image = iterative_object_removal(image, mask, reweighted_energy)
    cv2.imwrite(output_name, removed_image)
    return removed_image 

# Process the Cat image
removed_cat = process_image(cat_image, 'input/cat_mask.jpg', "output/removed_cat.jpg")

# Process the Sheep image
removed_sheep = process_image(sheep_image, "input/sheep_mask.jpg", "output/removed_sheep.jpg")

# Process the Dog image
removed_dog = process_image(dog_image, 'input/dog_mask.jpg', "output/removed_dog.jpg")



def restore_and_save_image(removed_image, original_width, output_name):
    """Restores the image to its original width and saves it."""
    restored_image = restore_image_to_original_size(removed_image, original_width)
    print(f"{output_name} Restored Shape: {restored_image.shape}")
    cv2.imwrite(output_name, restored_image)

# Save original widths
original_cat_width = cat_image.shape[1]
original_sheep_width = sheep_image.shape[1]
original_dog_width = dog_image.shape[1]

# Restore and save images
restore_and_save_image(removed_cat, original_cat_width, "output/final_cat.jpg")
restore_and_save_image(removed_sheep, original_sheep_width, "output/final_sheep.jpg")
restore_and_save_image(removed_dog, original_dog_width, "output/final_dog.jpg")

