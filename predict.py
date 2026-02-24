import cv2
import numpy as np
import pickle
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error loading image.")
        return None

    image = cv2.resize(image, (224, 224))
    norm = image / 255.0

    variance = np.var(norm)
    laplacian = cv2.Laplacian(norm, cv2.CV_64F)
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

    reduced = image // 16
    glcm = graycomatrix(
        reduced,
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

    features = np.hstack([
        variance,
        hf_variance,
        hist,
        contrast,
        energy,
        homogeneity,
        correlation,
        entropy
    ])

    return features


# Load trained model
model = pickle.load(open("svm_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Take user input
image_path = input("Enter image path: ")

features = extract_features(image_path)

if features is not None:
    features = scaler.transform([features])
    prediction = model.predict(features)[0]

    if prediction == 0:
        print("Prediction: REAL IMAGE")
    else:
        print("Prediction: AI GENERATED IMAGE")