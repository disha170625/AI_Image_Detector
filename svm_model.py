import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ===============================
# Load Dataset SAFELY
# ===============================
data = np.load("dataset.npy", allow_pickle=True)

# If dataset is empty
if len(data) == 0:
    print("Dataset is empty!")
    exit()

# Convert to list
data = list(data)

# Extract features and labels safely
X = []
y = []

for item in data:
    # If item is tuple (features, label)
    if isinstance(item, (list, tuple)) and len(item) == 2:
        X.append(item[0])
        y.append(item[1])
    else:
        print("Unexpected dataset format:", item)
        exit()

X = np.array(X)
y = np.array(y)

print("Total Samples:", len(X))
print("Feature Length:", len(X[0]))


# ===============================
# Scale Features
# ===============================
scaler = StandardScaler()
X = scaler.fit_transform(X)


# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# ===============================
# Train SVM
# ===============================
model = SVC(kernel='rbf', class_weight='balanced')
model.fit(X_train, y_train)


# ===============================
# Predict
# ===============================
y_pred = model.predict(X_test)


# ===============================
# Evaluation
# ===============================
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ===============================
# Save Model
# ===============================
pickle.dump(model, open("svm_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\nModel saved successfully.")