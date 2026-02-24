import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import train_test_split

# -----------------------------
# Feature Extraction Function
# -----------------------------
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    image = cv2.resize(image, (224, 224))
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

    return feature_vector


# -----------------------------
# Build Dataset
# -----------------------------
X = []
y = []

dataset_path = "dataset"

for label_folder in ["real", "ai"]:
    folder_path = os.path.join(dataset_path, label_folder)

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        features = extract_features(file_path)

        if features is not None:
            X.append(features)

            if label_folder == "real":
                y.append(0)
            else:
                y.append(1)

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))

# -----------------------------
# PROCESS 10: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))