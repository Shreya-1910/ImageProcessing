import cv2
import numpy as np
import os


# ---------------------------------------------------------------------------
# Paths  (all relative to the location of this script)
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

INPUT_BASE  = os.path.join(BASE_DIR, "001 - Input Images")
PIPE_BASE   = os.path.join(BASE_DIR, "002 - Image Processing Pipeline")
OUTPUT_BASE = os.path.join(BASE_DIR, "003 - Output Images")

DIFFICULTY_MAP = {
    "easy":   ("001 - Easy",   "001 - Easy"),
    "medium": ("002 - Medium", "002 - Medium"),
    "hard":   ("003 - Hard",   "003 - Hard"),
}

SUFFIX_MAP = {
    "easy":   ["Easy_1",   "Easy_2",   "Easy_3"],
    "medium": ["Medium_1", "Medium_2", "Medium_3"],
    "hard":   ["Hard_1",   "Hard_2",   "Hard_3"],
}


# ---------------------------------------------------------------------------
# Helper – make directory if it doesn't exist
# ---------------------------------------------------------------------------
def makedirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


# ---------------------------------------------------------------------------
# Core segmentation function
# ---------------------------------------------------------------------------
def segment_pmy(img: np.ndarray, pipeline_dir: str, stem: str):
    """
    Segment the PMY cell from *img*.

    Intermediate pipeline images are saved to pipeline_dir.
    Returns the segmented image (white background, only PMY visible).
    """

    # ── 1. Convert to LAB and split channels ──────────────────────────────
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_img)
    _save(pipeline_dir, stem + "_01_a_channel.jpg", a_channel)

    # ── 2. Gaussian blur on A channel ─────────────────────────────────────
    blur_imgA = cv2.GaussianBlur(a_channel, (5, 5), 15)
    _save(pipeline_dir, stem + "_02_blur_a.jpg", blur_imgA)

    # ── 3. Otsu's thresholding ────────────────────────────────────────────
    _, otsu_thresh = cv2.threshold(
        blur_imgA, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _save(pipeline_dir, stem + "_03_otsu.jpg", otsu_thresh)

    # ── 4. Morphological closing ──────────────────────────────────────────
    h, w = img.shape[:2]
    close_size = max(15, int(min(h, w) * 0.04))
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    closing = cv2.morphologyEx(
        otsu_thresh, cv2.MORPH_CLOSE, close_kernel, iterations=4
    )
    _save(pipeline_dir, stem + "_04_closing.jpg", closing)

    # ── 5. Flood-fill holes ───────────────────────────────────────────────
    flood_filled = closing.copy()
    fill_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_filled, fill_mask, (0, 0), 255)
    flood_fill_inverted = cv2.bitwise_not(flood_filled)
    closing = cv2.bitwise_or(closing, flood_fill_inverted)
    _save(pipeline_dir, stem + "_05_flood_fill.jpg", closing)

    # Guard: if mask is >90 % white the flood-fill over-segmented → revert
    if cv2.countNonZero(closing) > 0.9 * closing.size:
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closing = cv2.morphologyEx(
            otsu_thresh, cv2.MORPH_CLOSE, kernel_small, iterations=4
        )
        _save(pipeline_dir, stem + "_05b_closing_fallback.jpg", closing)

    # ── 6. Distance transform & sure foreground / background ─────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform, 0.2 * dist_transform.max(), 255, 0
    )
    sure_fg = cv2.erode(sure_fg.astype(np.uint8), kernel, iterations=1)
    sure_bg = cv2.dilate(closing, kernel, iterations=4)
    unknown = cv2.subtract(sure_bg, sure_fg)

    dist_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _save(pipeline_dir, stem + "_06_dist_transform.jpg", dist_vis)
    _save(pipeline_dir, stem + "_06b_sure_fg.jpg",       sure_fg)
    _save(pipeline_dir, stem + "_06c_sure_bg.jpg",       sure_bg)

    # ── 7. Sobel gradient magnitude ───────────────────────────────────────
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    _save(pipeline_dir, stem + "_07_gradient.jpg", gradient_magnitude)

    # ── 8. Marker labelling for watershed ────────────────────────────────
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # ── 9. Watershed ──────────────────────────────────────────────────────
    markers = cv2.watershed(
        cv2.cvtColor(gradient_magnitude, cv2.COLOR_GRAY2BGR), markers
    )

    # ── 10. Pick the best PMY segment (highest a-channel, penalise L) ────
    pmy_marker = None
    best_score = -np.inf

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
            pmy_marker  = label

    if pmy_marker is None:
        print(f"  [WARNING] No PMY marker found for {stem}. Returning white image.")
        return np.ones_like(img) * 255

    # ── 11. Build mask from best marker ──────────────────────────────────
    watershed_mask = np.zeros(closing.shape, dtype=np.uint8)
    watershed_mask[markers == pmy_marker] = 255

    # Keep only the largest contour
    contours, _ = cv2.findContours(
        watershed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        watershed_mask = np.zeros_like(watershed_mask)
        cv2.drawContours(
            watershed_mask, [largest], -1, 255, thickness=cv2.FILLED
        )
    _save(pipeline_dir, stem + "_08_watershed_mask.jpg", watershed_mask)

    # ── 12. Apply mask → white background ────────────────────────────────
    watershed_mask_binary = np.uint8(watershed_mask > 0) * 255
    extracted_img = cv2.bitwise_and(img, img, mask=watershed_mask_binary)
    extracted_img[watershed_mask_binary == 0] = [255, 255, 255]

    return extracted_img


# ---------------------------------------------------------------------------
# Utility: save an intermediate image
# ---------------------------------------------------------------------------
def _save(directory: str, filename: str, image: np.ndarray):
    os.makedirs(directory, exist_ok=True)
    cv2.imwrite(os.path.join(directory, filename), image)


# ---------------------------------------------------------------------------
# Discover all input images in a difficulty folder, sorted
# ---------------------------------------------------------------------------
def get_images(folder: str):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    )
    return files


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("PMY Segmentation Pipeline")
    print("=" * 60)

    for difficulty, (in_subfolder, out_subfolder) in DIFFICULTY_MAP.items():
        suffixes = SUFFIX_MAP[difficulty]

        input_folder = os.path.join(INPUT_BASE, in_subfolder)
        if not os.path.isdir(input_folder):
            print(f"\n[SKIP] Input folder not found: {input_folder}")
            continue

        image_files = get_images(input_folder)
        if not image_files:
            print(f"\n[SKIP] No images found in: {input_folder}")
            continue

        print(f"\n── {difficulty.upper()} ({len(image_files)} images) ──")

        for idx, fname in enumerate(image_files):
            if idx >= 3:
                print(f"  [INFO] More than 3 images found; processing first 3 only.")
                break

            suffix = suffixes[idx]                       # e.g. "Easy_1"
            stem   = os.path.splitext(fname)[0]          # filename without ext

            # Output file name: original-stem + suffix + .jpg
            out_fname = f"{stem}-{suffix}.jpg"

            # Paths
            in_path   = os.path.join(input_folder, fname)
            pipe_dir  = os.path.join(PIPE_BASE,   out_subfolder,
                                     f"00{idx+1} - {suffix}")
            out_dir   = os.path.join(OUTPUT_BASE, out_subfolder,
                                     f"00{idx+1} - {suffix}")
            out_path  = os.path.join(out_dir, out_fname)

            makedirs(pipe_dir, out_dir)

            print(f"  [{idx+1}] {fname}")

            # Read image
            img = cv2.imread(in_path)
            if img is None:
                print(f"       [ERROR] Cannot read image: {in_path}")
                continue

            # Save a copy of the original to the pipeline folder
            _save(pipe_dir, stem + "_00_original.jpg", img)

            # Segment
            result = segment_pmy(img, pipe_dir, stem)

            # Save output
            cv2.imwrite(out_path, result)
            print(f"       → Saved: {out_path}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()