import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


image = cv2.imread("proj1_gray.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found")
else:
    print("Image loaded successfully")

  
    image = image // 16  

  
    glcm = graycomatrix(image,
                        distances=[1],
                        angles=[0],
                        levels=16,
                        symmetric=True,
                        normed=True)


    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    
    glcm_matrix = glcm[:, :, 0, 0]
    entropy = -np.sum(glcm_matrix * np.log2(glcm_matrix + 1e-10))

    print("\nGLCM Features:")
    print("Contrast:", contrast)
    print("Energy:", energy)
    print("Homogeneity:", homogeneity)
    print("Correlation:", correlation)
    print("Entropy:", entropy)