import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# images = glob.glob("C:/Users/Navya/Downloads/Naturalize Dataset/*.jpg")
#
# for image_path in images:
img = cv2.imread("MMY 2K-PBC Train (26).png")

lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Split the LAB image into its channels
l_channel, a_channel, b_channel = cv2.split(lab_img)

# Apply Gaussian blur to the A channel
blur_imgA = cv2.GaussianBlur(a_channel, (5, 5), 10)

#Otsu's thresholding
_, otsu_thresh = cv2.threshold(blur_imgA, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphological operations
#Opening
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opening = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)

#Closing
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)#Connected components
num_labels, labels = cv2.connectedComponents(closing)

#Distance Transform
dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.9 * dist_transform.max(), 255, 0)
sure_fg = cv2.erode(sure_fg.astype(np.uint8), kernel, iterations=1)
sure_bg = cv2.dilate(closing, kernel, iterations=3)
unknown = cv2.subtract(sure_bg, sure_fg)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

#Marker labelling
num_labels, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

#Watershed
markers = cv2.watershed(cv2.cvtColor(gradient_magnitude, cv2.COLOR_GRAY2BGR), markers)

#Create a mask for the watershed result
watershed_mask = np.zeros_like(closing)
watershed_mask[markers > 1] = 255

# #Measuring size of components
# sizes = []
# for i in range(1, num_labels):
#     size = np.sum(labels == i)
#     sizes.append(size)
#
# #Find the largest component index, size[0] corresponds to label 1
# largest_component_index = np.argmax(sizes) + 1
#
# # Create a mask for the largest component
# largest_component_mask = np.zeros_like(closing)
# largest_component_mask[labels == largest_component_index] = 255

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot images
plt.figure(figsize=(12,8))

plt.subplot(2,5,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,5,2)
plt.imshow(l_channel, cmap='gray')
plt.title("L Channel")
plt.axis("off")

plt.subplot(2,5,3)
plt.imshow(a_channel, cmap='gray')
plt.title("A Channel")
plt.axis("off")

plt.subplot(2,5,4)
plt.imshow(b_channel, cmap='gray')
plt.title("B Channel")
plt.axis("off")

plt.subplot(2,5,5)
plt.imshow(blur_imgA, cmap='gray')
plt.title("Gaussian Blur (A Channel)")
plt.axis("off")

plt.subplot(2,5,6)
plt.imshow(otsu_thresh, cmap='gray')
plt.title("Otsu Threshold")
plt.axis("off")

plt.subplot(2,5,7)
plt.imshow(opening, cmap='gray')
plt.title("Opening")
plt.axis("off")

plt.subplot(2,5,8)
plt.imshow(closing, cmap='gray')
plt.title("Closing")
plt.axis("off")

plt.subplot(2,5,9)
plt.imshow(watershed_mask, cmap='gray')
plt.title("Watershed Mask")
plt.axis("off")

plt.tight_layout()
plt.show()