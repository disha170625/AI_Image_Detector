import cv2
import numpy as np

image = cv2.imread("proj1_gray.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found")
else:
    print("Image loaded successfully")

    image = image / 255.0

    variance = np.var(image)
    print("\nPixel Variance:", variance)

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    hf_variance = np.var(laplacian)

    print("High-Frequency Variance:", hf_variance)
    cv2.imwrite("laplacian.jpg", np.uint8(np.absolute(laplacian) * 255))
    print("Laplacian image saved as laplacian.jpg")