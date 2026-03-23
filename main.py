import cv2
import numpy as np
import matplotlib.pyplot as plt

BLUR_KERNEL = (5, 5)
DIST_THRESH = 0.2
MIN_AREA = 500

img = cv2.imread("EO 2K-PBC Train (423).jpg")
if img is None:
    raise ValueError("Image not found. Check file path.")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_channel, a_channel, b_channel = cv2.split(lab_img)

blur_imgA = cv2.GaussianBlur(a_channel, BLUR_KERNEL, 0)

_, otsu_thresh = cv2.threshold(
    blur_imgA, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

h, w = otsu_thresh.shape
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opening = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel_open)

# Adaptive closing based on image size
close_size = max(15, int(min(h, w) * 0.04))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close, iterations=3)

flood = closing.copy()
mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(flood, mask, (0, 0), 255)
flood_inv = cv2.bitwise_not(flood)
closing = cv2.bitwise_or(closing, flood_inv)

dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, DIST_THRESH * dist_transform.max(), 255, 0)
sure_fg = cv2.erode(np.uint8(sure_fg), kernel_open, iterations=1)

sure_bg = cv2.dilate(closing, kernel_open, iterations=4)
unknown = cv2.subtract(sure_bg, sure_fg)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
gradient = cv2.magnitude(sobel_x, sobel_y)
gradient = cv2.convertScaleAbs(gradient)
gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)

num_labels, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# Watershed
markers = cv2.watershed(cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR), markers)

best_score = -np.inf
best_marker = None
for label in range(2, markers.max()+1):
    region_mask = np.uint8(markers == label)
    area = cv2.countNonZero(region_mask)
    if area < MIN_AREA:
        continue

    mean_a = cv2.mean(a_channel, mask=region_mask)[0]
    mean_l = cv2.mean(l_channel, mask=region_mask)[0]

    score = mean_a - 0.5 * mean_l  # high a-channel, moderate brightness
    if score > best_score:
        best_score = score
        best_marker = label

watershed_mask = np.zeros_like(closing)
if best_marker is not None:
    watershed_mask[markers == best_marker] = 255

# Keep only largest connected component
contours, _ = cv2.findContours(watershed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    largest = max(contours, key=cv2.contourArea)
    watershed_mask = np.zeros_like(watershed_mask)
    cv2.drawContours(watershed_mask, [largest], -1, 255, -1)

final_mask = np.uint8(watershed_mask > 0) * 255

extracted_img = cv2.bitwise_and(img_rgb, img_rgb, mask=final_mask)
extracted_img[final_mask == 0] = [255, 255, 255]

plt.figure(figsize=(12,8))
titles = [
    "Original", "L Channel", "A Channel", "Blur",
    "Otsu", "Opening", "Closing", "Distance FG",
    "Watershed Mask", "Final Extraction"
]
images = [
    img_rgb, l_channel, a_channel, blur_imgA,
    otsu_thresh, opening, closing, sure_fg,
    watershed_mask, extracted_img
]

for i in range(len(images)):
    plt.subplot(2,5,i+1)
    if images[i].ndim == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()