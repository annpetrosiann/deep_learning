import cv2
import numpy as np
import copy

image = cv2.imread('car.webp')

if image is None:
    print("Could not open or find the image")


def display_image(name, image, save=False):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if save:
        cv2.imwrite(f'{name}.jpg', image)
        print(f"Image saved successfully")


#Resize image
resized_image = cv2.resize(image, (100, 100))
display_image('resized_image', resized_image, save=True)

#Flip the image vertically and horizontally.
flipped_vertically = cv2.flip(image, 0)
flipped_horizontally = cv2.flip(image, 1)
display_image('flipped_vertically', flipped_vertically, save=True)
display_image('flipped_horizontally', flipped_horizontally, save=True)

#Crop a specific region from the image.
cropped_image = image[50:100, 50:100]
display_image('cropped_image', cropped_image, save=True)

#Rotate the image by a certain angle.
rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
display_image('rotated_image', rotated_image, save=True)

#Convert the image to RGB, HSV, LAB and Grayscale color spaces.
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
display_image('rgb_image', rgb_image, save=True)
display_image('hsv_image', hsv_image, save=True)
display_image('lab_image', lab_image, save=True)
display_image('gray_image', gray_image, save=True)

# Draw basic shapes: lines, rectangles, and circles with specified parameters for position, dimensions, color,
# and thickness.
copied_image = copy.deepcopy(image)
cv2.line(copied_image, (50, 50), (200, 50), (0, 0, 255), 5)
cv2.rectangle(copied_image, (100, 100), (300, 200), (0, 255, 0), 3)
cv2.rectangle(copied_image, (350, 100), (500, 200), (255, 255, 0), -1)
cv2.circle(copied_image, (250, 300), 50, (255, 0, 0), 4)
cv2.circle(copied_image, (400, 300), 50, (0, 255, 255), -1)
display_image('copied_image', copied_image, save=True)

#Apply Gaussian Blur to reduce image noise and soften details.
#Apply a sharpening filter using a custom kernel to enhance edges and details.
#Apply median filtering to effectively remove salt-and-pepper noise.
#(Use appropriate photos to see the changes after applying the filters)


image_copy = image.copy()
gaussian_blur = cv2.GaussianBlur(image_copy, (15, 15), 0)
display_image('Gaussian Blur', gaussian_blur, save=True)

sharpening_kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])

sharpened = cv2.filter2D(image_copy, -1, sharpening_kernel)
display_image('Sharpened Image', sharpened, save=True)

median_filtered = cv2.medianBlur(image_copy, 5)
display_image('Median Filtered Image', median_filtered, save=True)

