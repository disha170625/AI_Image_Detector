import streamlit as st
import numpy as np
import joblib
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# -------------------------
# Load Model & Scaler
# -------------------------
@st.cache_resource
def load_files():
    model = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_files()

# -------------------------
# Feature Extraction (MATCH TRAINING EXACTLY)
# -------------------------
def extract_features(image):

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # MUST match training size
    image = cv2.resize(image, (224, 224))

    # Normalize
    norm = image / 255.0

    # ---------------- Noise Features ----------------
    variance = np.var(norm)

    laplacian = cv2.Laplacian(norm, cv2.CV_64F)
    hf_variance = np.var(laplacian)

    # ---------------- LBP ----------------
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

    # ---------------- GLCM ----------------
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

    # ---------------- Combine (EXACT SAME ORDER) ----------------
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

    return features.reshape(1, -1)


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(
    page_title="AI vs Real Image Detector",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ AI vs Real Image Detection")
st.write("Upload an image to detect whether it is AI-generated or Real.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Uploaded Image",
        use_container_width=True
    )

    # Extract features
    features = extract_features(image)

    # Scale
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]

    # IMPORTANT:
    # In your training:
    # 0 = real
    # 1 = ai
    if prediction == 0:
        st.success("✅ Real Image")
    else:
        st.error("⚠️ AI Generated Image")

    # Confidence
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features_scaled)
        confidence = np.max(prob) * 100
        st.write(f"Confidence: {confidence:.2f}%")