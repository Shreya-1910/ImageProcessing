import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# images = glob.glob("C:/Users/Navya/Downloads/Naturalize Dataset/*.jpg")
#
# for image_path in images:
img = cv2.imread("ERB 2K-PBC Train (106).jpg")

lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Split the LAB image into its channels
l_channel, a_channel, b_channel = cv2.split(lab_img)

# Apply Gaussian blur to the A channel
blur_imgA = cv2.GaussianBlur(a_channel, (5, 5), 15)

#Otsu's thresholding
_, otsu_thresh = cv2.threshold(blur_imgA, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphological operations
#Opening
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opening = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)

#Closing
h, w = img.shape[:2]
close_size = max(15, int(min(h, w) * 0.04))  # Adjust close size based on image dimensions
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel, iterations=4)#Connected components
num_labels, labels = cv2.connectedComponents(closing)

#Flood fill holes
flood_filled = closing.copy()
h, w = closing.shape[:2]
fill_mask = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(flood_filled, fill_mask, (0, 0), 255)
flood_fill_inverted = cv2.bitwise_not(flood_filled)
closing = cv2.bitwise_or(closing, flood_fill_inverted)

if cv2.countNonZero(closing) > 0.9 * closing.size:
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=4)

#Distance Transform
dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
sure_fg = cv2.erode(sure_fg.astype(np.uint8), kernel, iterations=1)
sure_bg = cv2.dilate(closing, kernel, iterations=4)
unknown = cv2.subtract(sure_bg, sure_fg)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

#Marker labelling
num_labels, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

#Watershed
markers = cv2.watershed(cv2.cvtColor(gradient_magnitude, cv2.COLOR_GRAY2BGR), markers)

#Pick only the PMY segment
watershed_mask = -1
best_score = -np.inf

for label in range(2, markers.max() + 1):
    region_mask = np.uint8(markers == label)
    area = cv2.countNonZero(region_mask)
    if area < 500:
        continue

    mean_intensity_a = cv2.mean(a_channel, mask=region_mask)[0]
    mean_intensity_l = cv2.mean(l_channel, mask=region_mask)[0]
    score = mean_intensity_a - 0.5 * mean_intensity_l
    if score > best_score:
        best_score = score
        pmy_marker = label

#Create a mask for the watershed result
watershed_mask = np.zeros_like(closing)
watershed_mask[markers == pmy_marker] = 255

#Keep only the largest connected component in the watershed mask
contours, _ = cv2.findContours(watershed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    watershed_mask = np.zeros_like(watershed_mask)
    cv2.drawContours(watershed_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

#Make sure mask is binary
watershed_mask_binary = np.uint8(watershed_mask > 0) * 255

#Use the watershed mask to segment the original image
white_bg = np.ones_like(img) * 255
extracted_img = cv2.bitwise_and(img, img, mask=watershed_mask_binary)
extracted_img[watershed_mask_binary == 0] = [255, 255, 255]

# Plot images
plt.figure(figsize=(12,8))

plt.subplot(2,5,1)
plt.imshow(img)
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

plt.subplot(2,5,10)
plt.imshow(extracted_img)
plt.title("Extracted Image")
plt.axis("off")

plt.tight_layout()
plt.show()