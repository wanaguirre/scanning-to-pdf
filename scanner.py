"""
scanner.py – Core image processing library.
Used by both scan.py (CLI) and app.py (web UI).
"""

import cv2
import numpy as np
from PIL import Image
import pillow_heif
import img2pdf
from pathlib import Path

pillow_heif.register_heif_opener()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif"}

# Paper sizes in mm (width × height, portrait).  'free' = keep warp dimensions.
PAPER_SIZES: dict[str, tuple[float, float] | None] = {
    "a4":     (210.0, 297.0),
    "letter": (215.9, 279.4),
    "a3":     (297.0, 420.0),
    "free":   None,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def order_corners(pts: np.ndarray) -> np.ndarray:
    pts  = pts.reshape(4, 2).astype(np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ], dtype=np.float32)


def _apply_exif_rotation(img: Image.Image) -> Image.Image:
    try:
        exif = img._getexif()
        if exif is None:
            return img
        orientation = exif.get(274)
        rotations = {3: 180, 6: 270, 8: 90}
        if orientation in rotations:
            return img.rotate(rotations[orientation], expand=True)
    except Exception:
        pass
    return img


def load_image(path: Path) -> tuple[np.ndarray, np.ndarray, Image.Image]:
    """Load image → (bgr, gray, pil). Handles HEIC/EXIF rotation."""
    pil = Image.open(path)
    pil = _apply_exif_rotation(pil)
    pil = pil.convert("RGB")
    bgr  = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray, pil


# ── Corner detection ───────────────────────────────────────────────────────────

def _auto_canny(gray: np.ndarray) -> np.ndarray:
    v  = np.median(gray)
    lo = int(max(0,   0.67 * v))
    hi = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lo, hi)
    return cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)


def _contour_corners(gray: np.ndarray) -> np.ndarray | None:
    h, w   = gray.shape
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq     = clahe.apply(gray)
    blurred = cv2.GaussianBlur(eq, (5, 5), 0)
    edges  = _auto_canny(blurred)
    edges  = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    best_area, best_quad = 0, None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        for eps in [0.01, 0.02, 0.03, 0.05]:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > 0.15 * h * w and area > best_area:
                    best_quad, best_area = approx, area
                break
    return order_corners(best_quad) if best_quad is not None else None


DETECT_MAX = 800   # max dimension used for corner detection (speed vs accuracy)


def detect_corners(path: Path) -> list[list[float]]:
    """
    Auto-detect the 4 document corners in TL, TR, BR, BL order.
    Detection runs on a downscaled copy for speed; corners are returned in
    original image pixel coordinates.
    Falls back to the full image corners if detection fails.
    """
    bgr, gray, _ = load_image(path)
    orig_h, orig_w = gray.shape

    # Downscale for detection
    det_scale = min(1.0, DETECT_MAX / max(orig_w, orig_h))
    if det_scale < 1.0:
        det_w     = int(orig_w * det_scale)
        det_h     = int(orig_h * det_scale)
        bgr_small  = cv2.resize(bgr,  (det_w, det_h), interpolation=cv2.INTER_AREA)
        gray_small = cv2.resize(gray, (det_w, det_h), interpolation=cv2.INTER_AREA)
    else:
        bgr_small, gray_small = bgr, gray

    # Simple contour-based detection on the downscaled image — fast enough for
    # interactive use. The UI lets the user drag corners to correct any mistakes.
    corners = _contour_corners(gray_small)

    if corners is None:
        corners = np.array([[0, 0], [orig_w, 0], [orig_w, orig_h], [0, orig_h]], dtype=np.float32)
    else:
        # Scale corners back to original image coordinates
        corners = corners / det_scale

    return corners.tolist()


# ── Processing ─────────────────────────────────────────────────────────────────

def perspective_warp(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = corners.astype(np.float32)
    W = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    H = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (W, H))


def resize_to_paper(img: np.ndarray, dpi: int, paper: str = "a4") -> np.ndarray:
    """Fit *img* (grayscale OR color BGR) onto the chosen paper canvas at *dpi*.
    paper can be 'a4', 'letter', 'a3', or 'free' (keep warp dimensions).
    """
    mm = PAPER_SIZES.get(paper.lower())
    if mm is None:          # 'free' – return the image unchanged
        return img
    h, w = img.shape[:2]
    color = img.ndim == 3
    # Auto-rotate canvas to match image orientation
    px_w = int(mm[0] / 25.4 * dpi)
    px_h = int(mm[1] / 25.4 * dpi)
    if w > h:               # landscape image → landscape canvas
        px_w, px_h = px_h, px_w
    scale = min(px_w / w, px_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas  = np.full((px_h, px_w, 3), 255, dtype=np.uint8) if color else np.full((px_h, px_w), 255, dtype=np.uint8)
    y_off   = (px_h - new_h) // 2
    x_off   = (px_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def enhance_scan(gray: np.ndarray) -> np.ndarray:
    """Black-and-white adaptive threshold (best for text/documents)."""
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21, C=10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def enhance_color(bgr: np.ndarray) -> np.ndarray:
    """Sharpen + mild CLAHE contrast boost for color scans."""
    # Unsharp mask
    blurred   = cv2.GaussianBlur(bgr, (0, 0), 3)
    sharpened = cv2.addWeighted(bgr, 1.4, blurred, -0.4, 0)
    # CLAHE on the L channel only (preserves color fidelity)
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def process_page(
    path: Path,
    corners: list[list[float]],
    dpi: int = 300,
    paper: str = "a4",
    mode: str = "bw",
) -> Image.Image:
    """
    Given an image path and 4 corners (in original pixel coords, TL/TR/BR/BL),
    return a processed PIL Image ready for PDF assembly.

    paper: 'a4' | 'letter' | 'a3' | 'free'
    mode:  'bw'    – black & white adaptive threshold (best for text)
           'gray'  – natural grayscale, no thresholding
           'color' – full colour with sharpening + contrast boost
    """
    bgr, gray, _ = load_image(path)
    corners_arr  = np.array(corners, dtype=np.float32)

    if mode == "color":
        img = perspective_warp(bgr, corners_arr)
        img = resize_to_paper(img, dpi, paper)
        img = enhance_color(img)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        img = perspective_warp(gray, corners_arr)
        img = resize_to_paper(img, dpi, paper)
        if mode == "bw":
            img = enhance_scan(img)
        # mode == 'gray': natural grayscale, no thresholding
        return Image.fromarray(img)


def build_pdf(pages: list[Image.Image], output_path: Path) -> None:
    """Save a list of PIL Images as a single PDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_paths = []
    tmp_dir   = output_path.parent / ".tmp_pages"
    tmp_dir.mkdir(exist_ok=True)

    for i, page in enumerate(pages):
        p = tmp_dir / f"page_{i:04d}.png"
        page.save(str(p), format="PNG")
        tmp_paths.append(str(p))

    with open(output_path, "wb") as f:
        f.write(img2pdf.convert(tmp_paths))

    for p in tmp_paths:
        Path(p).unlink(missing_ok=True)
    tmp_dir.rmdir()
