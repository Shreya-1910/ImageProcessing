import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('ERB 2K-PBC Train (23).jpg')

lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Split the LAB image into its channels
l_channel, a_channel, b_channel = cv2.split(lab_img)

# Apply Gaussian blur to the A channel
blur_imgA = cv2.GaussianBlur(a_channel, (3, 3), 0)

#Otsu's thresholding
_, otsu_thresh = cv2.threshold(blur_imgA, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphological operations
#Opening
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opening = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)

#Closing
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

#Connected components
cv2.connectedComponents(closing)
num_labels, labels = cv2.connectedComponents(closing)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot images
plt.figure(figsize=(12,8))

plt.subplot(2,4,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,4,2)
plt.imshow(l_channel, cmap='gray')
plt.title("L Channel")
plt.axis("off")

plt.subplot(2,4,3)
plt.imshow(a_channel, cmap='gray')
plt.title("A Channel")
plt.axis("off")

plt.subplot(2,4,4)
plt.imshow(b_channel, cmap='gray')
plt.title("B Channel")
plt.axis("off")

plt.subplot(2,4,5)
plt.imshow(blur_imgA, cmap='gray')
plt.title("Gaussian Blur (A Channel)")
plt.axis("off")

plt.subplot(2,4,6)
plt.imshow(otsu_thresh, cmap='gray')
plt.title("Otsu Threshold")
plt.axis("off")

plt.subplot(2,4,7)
plt.imshow(opening, cmap='gray')
plt.title("Opening")
plt.axis("off")

plt.subplot(2,4,8)
plt.imshow(closing, cmap='gray')
plt.title("Closing")
plt.axis("off")

plt.tight_layout()
plt.show()