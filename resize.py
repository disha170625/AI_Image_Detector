import cv2
image = cv2.imread("proj1.jpg")

if image is None:
    print("Error: Image not found")
else:
    print("Original Shape:", image.shape)

    resized_image = cv2.resize(image, (224, 224))

    print("Resized Shape:", resized_image.shape)

    cv2.imwrite("proj1_resized.jpg", resized_image)

    print("Resized image saved as proj1_resized.jpg")
    cv2.imshow("Resized Image", resized_image)
cv2.waitKey(2000)  
cv2.destroyAllWindows()