import cv2
image = cv2.imread("proj1_resized.jpg")

if image is None:
    print("Error: Image not found")
else:
    print("Original Shape:", image.shape)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Grayscale Shape:", gray_image.shape)

    cv2.imwrite("proj1_gray.jpg", gray_image)
    print("Grayscale image saved as proj1_gray.jpg")

  
    cv2.imshow("Grayscale Image", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()