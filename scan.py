"""
scan.py – Convert images in ./input to a scanned-document PDF in ./output.

Pipeline per image:
  1. Detect the document edges (largest 4-corner contour).
  2. Apply a perspective warp so the document fills the frame.
  3. Resize to A4 at 300 dpi (2480 × 3508 px).
  4. Enhance contrast: adaptive threshold → clean black-on-white document look.
  5. Assemble all pages into a single PDF.

Usage:
    python scan.py [--output scan.pdf] [--no-crop] [--dpi 300]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import img2pdf
import pillow_heif

# Register HEIC/HEIF support (iPhone photos)
pillow_heif.register_heif_opener()


# ── Constants ──────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif"}
A4_MM = (210, 297)          # width × height in millimetres

INPUT_DIR  = Path("input")
OUTPUT_DIR = Path("output")


# ── Helpers ────────────────────────────────────────────────────────────────────

def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners in [top-left, top-right, bottom-right, bottom-left] order."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s   = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],    # top-left
        pts[np.argmin(diff)], # top-right
        pts[np.argmax(s)],    # bottom-right
        pts[np.argmax(diff)], # bottom-left
    ], dtype=np.float32)


def _auto_canny(gray: np.ndarray) -> np.ndarray:
    """Canny with thresholds automatically derived from the pixel median."""
    v = np.median(gray)
    lo = int(max(0,   0.67 * v))
    hi = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lo, hi)
    return cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)


def _hough_corners(gray: np.ndarray) -> np.ndarray | None:
    """
    Detect document corners via Hough lines + k-means angle clustering.
    Much more robust than contours: works even when document edges are partially
    missing or have the same colour as the background.
    """
    h, w = gray.shape

    # CLAHE equalises uneven phone lighting before edge detection
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq      = clahe.apply(gray)
    blurred = cv2.GaussianBlur(eq, (5, 5), 0)
    edges   = _auto_canny(blurred)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
    if lines is None or len(lines) < 4:
        return None

    # Map theta to the unit circle so k-means handles the 0/π wrap correctly
    angles = np.array([l[0][1] for l in lines], dtype=np.float32)
    pts    = np.column_stack([np.cos(2 * angles), np.sin(2 * angles)]).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(pts, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()

    groups = [lines[labels == i] for i in (0, 1)]
    if any(len(g) == 0 for g in groups):
        return None

    # Compute intersections between every line in group 0 and group 1
    def _intersect(l1, l2):
        r1, t1 = l1[0]; r2, t2 = l2[0]
        A = np.array([[np.cos(t1), np.sin(t1)],
                      [np.cos(t2), np.sin(t2)]], dtype=np.float32)
        b = np.array([r1, r2], dtype=np.float32)
        try:
            x, y = np.linalg.solve(A, b)
            return (float(x), float(y))
        except np.linalg.LinAlgError:
            return None

    candidates = []
    for l1 in groups[0]:
        for l2 in groups[1]:
            pt = _intersect(l1, l2)
            # Accept only points within a loose margin around the image
            if pt and -w * 0.5 < pt[0] < w * 1.5 and -h * 0.5 < pt[1] < h * 1.5:
                candidates.append(pt)

    if len(candidates) < 4:
        return None

    # Cluster all candidate intersections down to 4 corners
    pts_arr = np.float32(candidates)
    _, labels2, centers = cv2.kmeans(
        pts_arr, 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    # Sanity-check: the bounding box of the 4 corners must cover ≥15% of the image
    xs, ys = centers[:, 0], centers[:, 1]
    area = (xs.max() - xs.min()) * (ys.max() - ys.min())
    if area < 0.15 * w * h:
        return None

    return order_corners(centers)


def _contour_corners(gray: np.ndarray) -> np.ndarray | None:
    """Fallback: CLAHE + auto-Canny + largest 4-point contour."""
    h, w = gray.shape
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq      = clahe.apply(gray)
    blurred = cv2.GaussianBlur(eq, (5, 5), 0)
    edges   = _auto_canny(blurred)
    edges   = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    best_area = 0
    best_quad = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        for eps in [0.01, 0.02, 0.03, 0.05]:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > 0.15 * h * w and area > best_area:
                    best_quad = approx
                    best_area = area
                break

    return order_corners(best_quad) if best_quad is not None else None


def _grabcut_corners(bgr: np.ndarray) -> np.ndarray | None:
    """Last resort: GrabCut foreground segmentation → largest contour → quad."""
    h, w = bgr.shape[:2]
    mx, my = w // 6, h // 6
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    mask     = np.zeros((h, w), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)

    doc_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    contours, _ = cv2.findContours(doc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt  = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    for eps in [0.01, 0.02, 0.03, 0.05]:
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4:
            return order_corners(approx)
    return None


def find_document_corners(gray: np.ndarray, bgr: np.ndarray) -> np.ndarray | None:
    """
    Three-stage corner detection, from most to least robust:
      1. Hough lines + k-means  (handles missing/partial edges, uneven lighting)
      2. CLAHE + auto-Canny + contour  (fast, good for high-contrast shots)
      3. GrabCut segmentation  (foreground/background, slowest, last resort)
    """
    corners = _hough_corners(gray)
    if corners is not None:
        return corners

    corners = _contour_corners(gray)
    if corners is not None:
        return corners

    corners = _grabcut_corners(bgr)
    return corners


def perspective_warp(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the image so that `corners` maps to a full-frame rectangle."""
    tl, tr, br, bl = corners

    width_top    = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left  = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    W = int(max(width_top,  width_bottom))
    H = int(max(height_left, height_right))

    dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (W, H))


def resize_to_a4(img: np.ndarray, dpi: int) -> np.ndarray:
    """
    Fit the image inside an A4 canvas at the given DPI without distortion.
    The image is scaled proportionally to fill as much of the page as possible,
    then centred on a white background.
    """
    px_w = int(A4_MM[0] / 25.4 * dpi)  # 2480 at 300 dpi
    px_h = int(A4_MM[1] / 25.4 * dpi)  # 3508 at 300 dpi

    h, w = img.shape[:2]

    # Choose portrait or landscape A4 based on image orientation
    if w > h:
        px_w, px_h = px_h, px_w  # landscape A4

    # Scale to fit inside the A4 bounds, preserving aspect ratio
    scale = min(px_w / w, px_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Place on a white A4 canvas, centred
    canvas = np.full((px_h, px_w), 255, dtype=np.uint8)
    y_off = (px_h - new_h) // 2
    x_off = (px_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas


def enhance_scan(gray: np.ndarray) -> np.ndarray:
    """
    Make the image look like a proper document scan:
      • Adaptive threshold → crisp black text on white background.
      • A gentle morphological close fills tiny holes inside letters.
    """
    # Adaptive threshold: handles uneven lighting well
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=10,
    )

    # Optionally close very small specks (comment out if you prefer raw threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


# ── Per-image processing ───────────────────────────────────────────────────────

def process_image(path: Path, dpi: int, do_crop: bool) -> Image.Image:
    """Full pipeline: load → crop → resize → enhance → return PIL Image."""
    # Load via Pillow so HEIC/HEIF (iPhone) and all other formats work uniformly
    pil_orig = Image.open(path)
    pil_orig = _apply_exif_rotation(pil_orig)
    pil_orig = pil_orig.convert("RGB")  # ensure no alpha / palette modes

    bgr  = cv2.cvtColor(np.array(pil_orig), cv2.COLOR_RGB2BGR)
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1. Detect & warp document
    if do_crop:
        corners = find_document_corners(gray, bgr)
        if corners is not None:
            bgr  = perspective_warp(bgr,  corners)
            gray = perspective_warp(gray, corners)
            print(f"  ✓ Document detected and cropped.")
        else:
            print(f"  ⚠  No clear document edge found – using full image.")
    else:
        print(f"  – Crop disabled, using full image.")

    # 3. Resize to A4
    gray = resize_to_a4(gray, dpi)

    # 4. Enhance contrast
    scanned = enhance_scan(gray)

    return Image.fromarray(scanned)


def _apply_exif_rotation(img: Image.Image) -> Image.Image:
    """Rotate image according to its EXIF orientation tag (if present)."""
    try:
        exif = img._getexif()
        if exif is None:
            return img
        orientation = exif.get(274)  # 274 = Orientation tag
        rotations = {3: 180, 6: 270, 8: 90}
        if orientation in rotations:
            return img.rotate(rotations[orientation], expand=True)
    except Exception:
        pass
    return img


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert scanned images to a PDF.")
    parser.add_argument("--output",   default="scan.pdf",
                        help="Output filename inside ./output  (default: scan.pdf)")
    parser.add_argument("--dpi",      type=int, default=300,
                        help="Target DPI for the A4 page (default: 300)")
    parser.add_argument("--no-crop",  action="store_true",
                        help="Skip document edge detection / perspective crop")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect & sort input images
    images = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        print(f"No images found in ./{INPUT_DIR}/.  Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)

    print(f"Found {len(images)} image(s).  DPI={args.dpi}  Crop={'off' if args.no_crop else 'on'}\n")

    processed_paths: list[str] = []
    tmp_dir = OUTPUT_DIR / ".tmp_scan_pages"
    tmp_dir.mkdir(exist_ok=True)

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}")
        try:
            page = process_image(img_path, dpi=args.dpi, do_crop=not args.no_crop)
            out_path = tmp_dir / f"page_{i:04d}.png"
            page.save(str(out_path), format="PNG")
            processed_paths.append(str(out_path))
        except Exception as exc:
            print(f"  ✗ Error: {exc}")

    if not processed_paths:
        print("\nNo pages were processed successfully.")
        sys.exit(1)

    # Assemble PDF
    output_pdf = OUTPUT_DIR / args.output
    with open(output_pdf, "wb") as f:
        f.write(img2pdf.convert(processed_paths))

    # Clean up temp pages
    for p in processed_paths:
        Path(p).unlink(missing_ok=True)
    tmp_dir.rmdir()

    print(f"\nDone! PDF written to: {output_pdf}  ({len(processed_paths)} page(s))")


if __name__ == "__main__":
    main()
