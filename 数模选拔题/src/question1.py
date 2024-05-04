import cv2
import numpy as np
from PIL import Image

def load_images(file_paths):
    """ Load images from specified file paths using OpenCV. """
    return [cv2.imread(fp, cv2.IMREAD_GRAYSCALE) for fp in file_paths]

def concatenate_images(images, order):
    """ Concatenate images in the specified order using PIL. """
    images_pil = [Image.fromarray(img) for img in images]  # Convert to PIL images for easier handling
    total_width = sum(images_pil[idx-1].width for idx in order)  # Sum up the widths of images in the given order
    max_height = max(images_pil[idx-1].height for idx in order)  # Get the max height among all images
    final_img = Image.new('L', (total_width, max_height))  # Create a new image with enough width and max height

    x_offset = 0
    for idx in order:
        final_img.paste(images_pil[idx-1], (x_offset, 0))  # Paste each image at the correct offset
        x_offset += images_pil[idx-1].width  # Update the offset for the next image

    return final_img

# Paths for images, assuming images are named as 1.bmp to 15.bmp and stored in the /mnt/data/ directory
file_paths = [f'question1/{i}.bmp' for i in range(1, 16)]
images = load_images(file_paths)
order = [6, 3, 11, 7, 14, 8, 5, 15, 1, 2, 4, 13, 9, 10, 12]  # The specific order you provided
final_image = concatenate_images(images, order)
final_image.show()  # Display the final concatenated image
