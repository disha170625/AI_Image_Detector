import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


image = cv2.imread("proj1_gray.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found")
else:
    print("Image loaded successfully")

    norm_image = image / 255.0

  
    variance = np.var(norm_image)

    laplacian = cv2.Laplacian(norm_image, cv2.CV_64F)
    hf_variance = np.var(laplacian)

 
    radius = 1
    n_points = 8 * radius

    lbp = local_binary_pattern(image, n_points, radius, method="uniform")

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)


    reduced_image = image // 16

    glcm = graycomatrix(
        reduced_image,
        distances=[1],
        angles=[0],
        levels=16,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    glcm_matrix = glcm[:, :, 0, 0]
    entropy = -np.sum(glcm_matrix * np.log2(glcm_matrix + 1e-10))

    
    feature_vector = np.hstack([
        variance,
        hf_variance,
        hist,
        contrast,
        energy,
        homogeneity,
        correlation,
        entropy
    ])

    print("\nFinal Feature Vector:")
    print(feature_vector)

    print("\nFeature Vector Length:", len(feature_vector))