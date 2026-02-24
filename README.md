# AI Image Detector

A machine learning web application that classifies whether an image is AI-generated or a real photograph using classical texture-based feature extraction and an SVM classifier.

---

## Features
- Grayscale preprocessing and resizing
- Texture feature extraction (LBP, GLCM)
- Noise variance analysis
- Support Vector Machine (SVM) classifier
- Streamlit web interface
- Confidence score display

---

## Tech Stack
- Python
- OpenCV
- NumPy
- Scikit-learn
- Scikit-image
- Streamlit

---


---

## Dataset

The dataset consists of real and AI-generated landscape images.

**Sources:**
- Real images: Unsplash
- AI images: Generated using AI image generation tools

> Note: The full dataset is not included in this repository due to size and licensing constraints.

###Install dependencies

### Train the model

### Run the web application

##How It Works

1. The uploaded image is converted to grayscale and resized to 224x224.
2. Texture features are extracted using:
   - Local Binary Pattern (LBP)
   - Gray-Level Co-occurrence Matrix (GLCM)
   - Noise variance and Laplacian variance
3. Features are standardized using a scaler.
4. A trained Support Vector Machine (SVM) classifier predicts the class.
5. The result is displayed on the web interface with a confidence score.
   
## Model Performance

The SVM classifier achieved approximately **77–80% accuracy** on the curated dataset.

Performance may vary depending on dataset diversity and image quality.

## Project Objective

This project demonstrates:

- Classical machine learning for image classification
- Feature engineering techniques in computer vision
- Model training and evaluation
- Deployment using Streamlit
- End-to-end ML pipeline development

## Future Improvements

- Increase dataset size for better generalization
- Experiment with deep learning models (CNNs)
- Improve confidence calibration
- Deploy online for public access
