import os
import cv2
import numpy as np

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# =====================================
# Feature Extraction Function
# =====================================
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    image = cv2.resize(image, (224, 224))
    norm = image / 255.0

    # Noise Features
    variance = np.var(norm)
    laplacian = cv2.Laplacian(norm, cv2.CV_64F)
    hf_variance = np.var(laplacian)

    # LBP
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

    # GLCM
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

    return np.hstack([
        variance,
        hf_variance,
        hist,
        contrast,
        energy,
        homogeneity,
        correlation,
        entropy
    ])


# =====================================
# Build Dataset
# =====================================
X = []
y = []

dataset_path = "dataset"

for label, folder in enumerate(["real", "ai"]):
    folder_path = os.path.join(dataset_path, folder)

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        features = extract_features(file_path)

        if features is not None:
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Total Samples:", len(X))


# =====================================
# Scale Features
# =====================================
scaler = StandardScaler()
X = scaler.fit_transform(X)


# =====================================
# Train-Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# =====================================
# Train SVM
# =====================================
model = SVC(kernel='rbf', class_weight='balanced')
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# =====================================
# Evaluate
# =====================================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
import pickle

pickle.dump(model, open("svm_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model saved successfully.")