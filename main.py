import cv2
import numpy as np
import matplotlib.pyplot as plt

BLUR_KERNEL = (5,5)
MEDIAN_KERNEL = 5
DIST_THRESH = 0.2
MIN_AREA = 500


img = cv2.imread("EO 2K-PBC Train (39).jpg")

if img is None:
    raise ValueError("Image not found. Check file path.")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# colour space
lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_channel, a_channel, b_channel = cv2.split(lab_img)


#Clahe
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
a_channel = clahe.apply(a_channel)

# gaussian smooth

blur_imgA = cv2.GaussianBlur(a_channel, BLUR_KERNEL, 0)

# median filtering
median_img = cv2.medianBlur(blur_imgA, MEDIAN_KERNEL)

#otsu threshold
_, otsu_thresh = cv2.threshold(
    median_img, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# morphological

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

opening = cv2.morphologyEx(
    otsu_thresh,
    cv2.MORPH_OPEN,
    kernel
)

closing = cv2.morphologyEx(
    opening,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=3
)

#hole-filling
flood = closing.copy()
h, w = closing.shape
mask = np.zeros((h+2, w+2), np.uint8)

cv2.floodFill(flood, mask, (0,0), 255)
flood_inv = cv2.bitwise_not(flood)

closing = cv2.bitwise_or(closing, flood_inv)


dist_transform = cv2.distanceTransform(
    closing,
    cv2.DIST_L2,
    5
)

_, sure_fg = cv2.threshold(
    dist_transform,
    DIST_THRESH * dist_transform.max(),
    255,
    0
)

sure_fg = np.uint8(sure_fg)

sure_bg = cv2.dilate(closing, kernel, iterations=3)

unknown = cv2.subtract(sure_bg, sure_fg)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)

gradient = cv2.magnitude(sobel_x, sobel_y)
gradient = cv2.convertScaleAbs(gradient)

gradient = cv2.normalize(
    gradient,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

num_labels, markers = cv2.connectedComponents(sure_fg)

markers = markers + 1
markers[unknown == 255] = 0

#watershed
markers = cv2.watershed(
    cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR),
    markers
)

best_score = -np.inf
best_marker = None

for label in range(2, markers.max()+1):

    region_mask = np.uint8(markers == label)

    area = cv2.countNonZero(region_mask)

    if area < MIN_AREA:
        continue

    mean_a = cv2.mean(a_channel, mask=region_mask)[0]
    mean_l = cv2.mean(l_channel, mask=region_mask)[0]

    score = mean_a - 0.5 * mean_l

    if score > best_score:
        best_score = score
        best_marker = label

#mask
watershed_mask = np.zeros_like(closing)

if best_marker is not None:
    watershed_mask[markers == best_marker] = 255


#contour filt

contours, _ = cv2.findContours(
    watershed_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if contours:

    largest = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)

    circularity = 4*np.pi*area/(perimeter*perimeter+1e-6)

    if circularity > 0.3:

        largest = cv2.approxPolyDP(
            largest,
            0.01*perimeter,
            True
        )

        watershed_mask = np.zeros_like(watershed_mask)

        cv2.drawContours(
            watershed_mask,
            [largest],
            -1,
            255,
            -1
        )
#extraction
final_mask = np.uint8(watershed_mask > 0)*255

extracted_img = cv2.bitwise_and(img_rgb, img_rgb, mask=final_mask)

white_bg = np.ones_like(img_rgb)*255
extracted_img[final_mask == 0] = [255,255,255]

plt.figure(figsize=(12,8))

titles = [
    "Original",
    "L Channel",
    "A Channel",
    "Blur",
    "Median",
    "Otsu",
    "Opening",
    "Closing",
    "Watershed Mask",
    "Final Extraction"
]

images = [
    img_rgb,
    l_channel,
    a_channel,
    blur_imgA,
    median_img,
    otsu_thresh,
    opening,
    closing,
    watershed_mask,
    extracted_img
]

for i in range(len(images)):

    plt.subplot(2,5,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()