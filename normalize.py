import cv2
import numpy as np

image = cv2.imread("proj1_gray.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found")
else:
    print("Before Normalization:")
    print("Min pixel value:", np.min(image))
    print("Max pixel value:", np.max(image))

   
    normalized_image = image / 255.0

    print("\nAfter Normalization:")
    print("Min pixel value:", np.min(normalized_image))
    print("Max pixel value:", np.max(normalized_image))

    print("\nData type:", normalized_image.dtype)