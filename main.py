import cv2
import numpy as np
import glob
import os
from scipy.spatial.distance import directed_hausdorff

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

# Hausdorff Distance
def calculate_hausdorff(mask1, mask2):
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    pts1 = np.column_stack(np.where(mask1 > 0))
    pts2 = np.column_stack(np.where(mask2 > 0))
    if len(pts1) == 0 or len(pts2) == 0:
        return np.nan
    d1 = directed_hausdorff(pts1, pts2)[0]
    d2 = directed_hausdorff(pts2, pts1)[0]
    return max(d1, d2)

# Precision
def calculate_precision(mask1, mask2):
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    tp = np.sum(mask1 & mask2)
    fp = np.sum(mask1 & (~mask2))
    if tp + fp == 0:
        return 1.0
    return tp / (tp + fp)

# Recall
def calculate_recall(mask1, mask2):
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    tp = np.sum(mask1 & mask2)
    fn = np.sum((~mask1) & mask2)
    if tp + fn == 0:
        return 1.0
    return tp / (tp + fn)

# Cell Count Accuracy
def calculate_cell_count_accuracy(pred_mask, gt_mask):
    # Count number of connected components (cells)
    num_pred, _ = cv2.connectedComponents((pred_mask > 0).astype(np.uint8))
    num_gt, _ = cv2.connectedComponents((gt_mask > 0).astype(np.uint8))
    # Subtract background component
    num_pred -= 1
    num_gt -= 1
    if num_gt == 0:
        return 1.0 if num_pred == 0 else 0.0
    return 1 - abs(num_pred - num_gt) / num_gt

# Store metrics
dice_scores, iou_scores, hausdorff_scores = [], [], []
precision_scores, recall_scores, cellcount_scores = [], [], []

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

    # Metrics
    dice = calculate_dice(pred_bin, gt_bin)
    iou = calculate_iou(pred_bin, gt_bin)
    hausdorff = calculate_hausdorff(pred_bin, gt_bin)
    precision = calculate_precision(pred_bin, gt_bin)
    recall = calculate_recall(pred_bin, gt_bin)
    cellcount_acc = calculate_cell_count_accuracy(pred_bin, gt_bin)

    dice_scores.append((img_name, dice))
    iou_scores.append((img_name, iou))
    hausdorff_scores.append((img_name, hausdorff))
    precision_scores.append((img_name, precision))
    recall_scores.append((img_name, recall))
    cellcount_scores.append((img_name, cellcount_acc))

    print(f"{img_name}: Dice={dice:.4f}, IoU={iou:.4f}, Hausdorff={hausdorff:.4f}, "
          f"Precision={precision:.4f}, Recall={recall:.4f}, CellCountAcc={cellcount_acc:.4f}")

# Final Results
def print_avg_and_top3(scores, metric_name, reverse=True):
    avg_score = np.nanmean([score for _, score in scores])
    print(f"\nAverage {metric_name}: {avg_score:.4f}")
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=reverse)
    print(f"Top 3 {metric_name}:")
    for name, score in sorted_scores[:3]:
        print(f"{name}: {score:.4f}")

print_avg_and_top3(dice_scores, "Dice Score")
print_avg_and_top3(iou_scores, "IoU Score")
print_avg_and_top3(precision_scores, "Precision")
print_avg_and_top3(recall_scores, "Recall")
print_avg_and_top3(cellcount_scores, "Cell Count accuracy")
print_avg_and_top3(hausdorff_scores, "Hausdorff Distance", reverse=False)