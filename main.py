import cv2
import numpy as np
import glob
import os

# Dataset folders

img_folder = r"C:\Users\Shrey\PycharmProjects\PythonProject\ERB"
mask_folder = r"C:\Users\Shrey\PycharmProjects\PythonProject\ERBGroundtruth"


# Dice Score
def calculate_dice(mask1, mask2):
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)

    intersection = np.sum(mask1 & mask2)
    total_sum = np.sum(mask1) + np.sum(mask2)

    if total_sum == 0:
        return 1.0

    return (2.0 * intersection) / total_sum

# IoU Function

def calculate_iou(mask1, mask2):
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)

    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)

    if union == 0:
        return 1.0

    return intersection / union

dice_scores = []
iou_scores = []

image_files = glob.glob(os.path.join(img_folder, "*.jpg"))
print("Total images found:", len(image_files))

for img_path in image_files:

    img_name = os.path.basename(img_path).replace(".jpg", "")
    mask_path = os.path.join(mask_folder, img_name + "_mask.png")

    img = cv2.imread(img_path)
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue
    if gt_mask is None:
        print(f"{img_name}: no ground truth mask")
        continue

    # Convert to LAB
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_img)

    blur_imgA = cv2.GaussianBlur(a_channel, (5, 5), 15)

    # Otsu Threshold
    _, otsu_thresh = cv2.threshold(
        blur_imgA, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    h, w = img.shape[:2]
    close_size = max(15, int(min(h, w) * 0.04))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))

    closing = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, close_kernel, iterations=4)

    # Flood fill holes
    flood_filled = closing.copy()
    fill_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_filled, fill_mask, (0, 0), 255)
    flood_fill_inverted = cv2.bitwise_not(flood_filled)
    closing = cv2.bitwise_or(closing, flood_fill_inverted)

    if cv2.countNonZero(closing) > 0.9 * closing.size:
        closing = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

    # Distance transform
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = cv2.erode(sure_fg.astype(np.uint8), kernel, iterations=1)
    sure_bg = cv2.dilate(closing, kernel, iterations=4)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Gradient magnitude
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

    # Marker labelling
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed segmentation
    markers = cv2.watershed(cv2.cvtColor(gradient_magnitude, cv2.COLOR_GRAY2BGR), markers)

    best_score = -np.inf
    pmy_marker = -1

    # Choose best segment
    for label in range(2, markers.max() + 1):
        region_mask = np.uint8(markers == label)
        area = cv2.countNonZero(region_mask)
        if area < 500:
            continue

        mean_a = cv2.mean(a_channel, mask=region_mask)[0]
        mean_l = cv2.mean(l_channel, mask=region_mask)[0]
        score = mean_a - 0.5 * mean_l

        if score > best_score:
            best_score = score
            pmy_marker = label

    watershed_mask = np.zeros_like(closing)
    if pmy_marker != -1:
        watershed_mask[markers == pmy_marker] = 255

    # Keep largest component
    contours, _ = cv2.findContours(watershed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        watershed_mask = np.zeros_like(watershed_mask)
        cv2.drawContours(watershed_mask, [largest], -1, 255, thickness=cv2.FILLED)

    watershed_mask_binary = (np.uint8(watershed_mask > 0) * 255)

    # Threshold
    _, pred_bin = cv2.threshold(watershed_mask_binary, 127, 255, cv2.THRESH_BINARY)
    _, gt_bin = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)

    # Dice & IoU
    dice = calculate_dice(pred_bin, gt_bin)
    iou = calculate_iou(pred_bin, gt_bin)

    dice_scores.append((img_name, dice))
    iou_scores.append((img_name, iou))

    print(f"{img_name}: Dice={dice:.4f}, IoU={iou:.4f}")

# Final Results
if dice_scores:

    avg_dice = np.mean([score for _, score in dice_scores])
    avg_iou = np.mean([score for _, score in iou_scores])

    print("\nAverage Dice Score:", avg_dice)
    print("Average IoU Score:", avg_iou)

    sorted_dice = sorted(dice_scores, key=lambda x: x[1], reverse=True)
    sorted_iou = sorted(iou_scores, key=lambda x: x[1], reverse=True)

    print("\nTop 3 Dice Scores:")
    for name, score in sorted_dice[:3]:
        print(f"{name}: {score:.4f}")

    print("\nTop 3 IoU Scores:")
    for name, score in sorted_iou[:3]:
        print(f"{name}: {score:.4f}")