import cv2

image = cv2.imread("proj2.jpg")

if image is None:
    print("Error: Image not found!")
else:
    print("Image loaded successfully!")
    print("Image Shape:", image.shape)