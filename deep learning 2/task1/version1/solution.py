import cv2 as cv
import numpy as np

# Load an image
image = cv.imread('nature.jpg')
if image is None:
    print("Could not open or find the image")

# Display and save the image
def show_and_save(description, image):
    cv.imshow(description,image)
    a = cv.waitKey(0)
    if a == ord('a'):
        cv.destroyAllWindows()
    filename = f'/Users/annpetrosiann/Desktop/{description}.jpg'
    cv.imwrite(filename, image)


# Resize the image
resized_image = cv.resize(image, (400, 500))
show_and_save( "Resized image",resized_image)

# Flip the Image
flipped_horizontal = cv.flip(image, 1)
show_and_save( "Flipped horizontal image",flipped_horizontal)

flipped_vertical = cv.flip(image, 0)
show_and_save( "Flipped vertical image",flipped_vertical)

# Crop the image
cropped_image = image[100:500, 100:500]
show_and_save( "Cropped image",cropped_image)

#Rotate the image
height, width = image.shape[:2]
center = (width / 2, height / 2)
rotation_matrix = cv.getRotationMatrix2D(center, 45, 1.0)
rotated_image = cv.warpAffine(image, rotation_matrix, (width, height))
show_and_save( "Rotated",rotated_image)

# Convert to different color spaces
rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
show_and_save("RGB Image", rgb_image)

hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
show_and_save("HSV Image", hsv_image)

lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
show_and_save("LAB Image", lab_image)

gray_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
show_and_save( "Gray image",gray_image)

# Apply image filters
Gaussian = cv.GaussianBlur(image, (5, 5),0)
show_and_save("Gaussian", Gaussian)

sharpened_image = cv.filter2D(image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
show_and_save("Sharpened image", sharpened_image)

median_filtered_image = cv.medianBlur(image, 5)
show_and_save("Median filtered image", median_filtered_image)


# Draw shapes
cv.line(image, (50, 30), (200, 30), (255, 0, 0), 5)  # Blue line
cv.rectangle(image, (50, 60), (200, 120), (0, 255, 0), 3)  # Green rectangle
cv.circle(image, (150, 150), 40, (0, 0, 255), -1)  # Red circle
show_and_save("Shapes on Image", image)


