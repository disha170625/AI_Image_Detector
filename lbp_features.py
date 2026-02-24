import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Load grayscale image
image = cv2.imread("proj1_gray.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found")
else:
    print("Image loaded successfully")

   
    radius = 1
    n_points = 8 * radius

    
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")

    
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, n_points + 3),
                           range=(0, n_points + 2))

  
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    print("\nLBP Feature Vector:")
    print(hist)
    print("\nFeature Vector Length:", len(hist))

    
    cv2.imwrite("lbp_image.jpg", np.uint8(lbp))
    print("\nLBP image saved as lbp_image.jpg")