import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("rose-grayscale.jpg", cv2.IMREAD_GRAYSCALE)

# Task 1: Erosion and Dilation
kernel = np.ones((5, 5), np.uint8)
erosion_1 = cv2.erode(image, kernel, iterations=1)
erosion_5 = cv2.erode(image, kernel, iterations=5)
erosion_10 = cv2.erode(image, kernel, iterations=10)

dilation_1 = cv2.dilate(image, kernel, iterations=1)
dilation_5 = cv2.dilate(image, kernel, iterations=5)
dilation_10 = cv2.dilate(image, kernel, iterations=10)

plt.figure(figsize=(18, 12))
images = [
    image,
    erosion_1,
    erosion_5,
    erosion_10,
    dilation_1,
    dilation_5,
    dilation_10,
]
titles = [
    "Original",
    "Erosion 1",
    "Erosion 5",
    "Erosion 10",
    "Dilation 1",
    "Dilation 5",
    "Dilation 10",
]
for i in range(7):
    plt.subplot(2, 4, i + 1), plt.imshow(images[i], "gray")
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# Task 2: Histogram Equalization
image_eq = cv2.imread("Grayscale.jpg", cv2.IMREAD_GRAYSCALE)
equalized_image = cv2.equalizeHist(image_eq)

plt.figure(figsize=(10, 7))
plt.subplot(221), plt.hist(image_eq.ravel(), 256, (0, 256))
plt.title("Original Histogram")
plt.subplot(222), plt.hist(equalized_image.ravel(), 256, (0, 256))
plt.title("Equalized Histogram")
plt.subplot(223), plt.imshow(image_eq, "gray"), plt.title("Original Image")
plt.subplot(224), plt.imshow(equalized_image, "gray"), plt.title(
    "Equalized Image"
)
plt.show()

# Task 3: Edge Detection
edges_50_150 = cv2.Canny(image, 50, 150)
edges_100_200 = cv2.Canny(image, 100, 200)

sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel_combined = cv2.magnitude(sobelx, sobely)
_, sobel_edges = cv2.threshold(sobel_combined, 100, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(10, 7))
plt.subplot(2, 2, 1), plt.imshow(edges_50_150, "gray"), plt.title(
    "Canny 50-150"
)
plt.subplot(2, 2, 2), plt.imshow(edges_100_200, "gray"), plt.title(
    "Canny 100-200"
)
plt.subplot(2, 2, 3), plt.imshow(sobel_combined, "gray"), plt.title(
    "Sobel Combined"
)
plt.subplot(2, 2, 4), plt.imshow(sobel_edges, "gray"), plt.title("Sobel Edges")
plt.show()

# Task 4: Feature Extraction (HOG and SIFT)
hog = cv2.HOGDescriptor()
hog_features = hog.compute(image)
print("HOG Features shape:", hog_features.shape)
print("First 10 HOG Features:", hog_features[:10])

sift = cv2.SIFT_create()
keypoints1, _ = sift.detectAndCompute(image, None)
keypoints2, _ = sift.detectAndCompute(image_eq, None)
image_with_keypoints1 = cv2.drawKeypoints(
    image, keypoints1, None, (255, 0, 0), 4
)
image_with_keypoints2 = cv2.drawKeypoints(
    image_eq, keypoints2, None, (255, 0, 0), 4
)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_with_keypoints1)
plt.title(f"Image 1: {len(keypoints1)} Keypoints")
plt.subplot(1, 2, 2)
plt.imshow(image_with_keypoints2)
plt.title(f"Image 2: {len(keypoints2)} Keypoints")
plt.show()

print(f"Number of keypoints in Image 1: {len(keypoints1)}")
print(f"Number of keypoints in Image 2: {len(keypoints2)}")