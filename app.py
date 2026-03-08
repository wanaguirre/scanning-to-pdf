"""
app.py – Interactive web UI for scanning documents to PDF.
Run with: python app.py
Then open http://localhost:5000 in your browser.
"""

import io
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

from scanner import (
    SUPPORTED_EXTENSIONS,
    detect_corners,
    load_image,
    process_page,
    build_pdf,
)

app = Flask(__name__)

INPUT_DIR  = Path("input")
OUTPUT_DIR = Path("output")
DISPLAY_MAX = 3000  # max pixel dimension when serving images to the browser
THUMB_MAX   = 320   # thumbnail size for home-page grid


def _list_images() -> list[Path]:
    return sorted(
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _pil_to_jpeg_b64(img: Image.Image, quality: int = 92) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _serve_display_image(path: Path) -> Image.Image:
    """Load and downscale to DISPLAY_MAX for fast browser rendering."""
    _, _, pil = load_image(path)
    w, h = pil.size
    if max(w, h) > DISPLAY_MAX:
        scale = DISPLAY_MAX / max(w, h)
        pil   = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/outputs")
def api_outputs():
    """List previously generated PDFs in output/."""
    pdfs = sorted(OUTPUT_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return jsonify({"files": [
        {"name": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime}
        for p in pdfs
    ]})


@app.route("/output/<name>")
def serve_output(name: str):
    """Download a previously generated PDF."""
    path = (OUTPUT_DIR / name).resolve()
    if not str(path).startswith(str(OUTPUT_DIR.resolve())) or not path.exists():
        return "Not found", 404
    return send_file(str(path), mimetype="application/pdf", as_attachment=True, download_name=name)


@app.route("/api/thumb/<name>")
def api_thumb(name: str):
    """Serve a tiny JPEG thumbnail for the home-page image grid."""
    path = INPUT_DIR / name
    if not path.exists() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return jsonify({"error": "not found"}), 404
    _, _, pil = load_image(path)
    w, h = pil.size
    scale = THUMB_MAX / max(w, h)
    if scale < 1.0:
        pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=82)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/api/images")
def api_images():
    """List all input images and auto-detect their corners."""
    images = _list_images()
    result = []
    for p in images:
        try:
            _, _, pil = load_image(p)
            orig_w, orig_h = pil.size
            corners = detect_corners(p)
            result.append({
                "name":   p.name,
                "width":  orig_w,
                "height": orig_h,
                "corners": corners,    # [[x,y] x4] in original pixel coords
            })
        except Exception as e:
            result.append({"name": p.name, "error": str(e)})
    return jsonify({"images": result})


@app.route("/api/image/<name>")
def api_image(name: str):
    """Serve a display-resolution JPEG of the requested image."""
    path = INPUT_DIR / name
    if not path.exists() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return jsonify({"error": "not found"}), 404
    pil  = _serve_display_image(path)
    buf  = io.BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/api/preview", methods=["POST"])
def api_preview():
    """
    Body: { name, corners [[x,y]x4] in original coords }
    Returns: { preview: "<base64 JPEG>" } of the warped+enhanced result.
    """
    data    = request.get_json()
    path    = INPUT_DIR / data["name"]
    corners = data["corners"]
    paper   = data.get("paper", "a4")
    mode    = data.get("mode", "bw")

    page = process_page(path, corners, dpi=300, paper=paper, mode=mode)

    return jsonify({"preview": _pil_to_jpeg_b64(page, quality=92)})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Accept one or more image files and save them to input/."""
    files  = request.files.getlist("files")
    saved  = []
    errors = []
    for f in files:
        suffix = Path(f.filename).suffix.lower()
        if suffix in SUPPORTED_EXTENSIONS:
            dest = INPUT_DIR / f.filename
            f.save(str(dest))
            saved.append(f.filename)
        else:
            errors.append(f"{f.filename} (unsupported format)")
    return jsonify({"saved": saved, "errors": errors})


@app.route("/api/image/<name>", methods=["DELETE"])
def api_delete_image(name: str):
    """Remove an image from input/."""
    path = INPUT_DIR / name
    if path.exists() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
        path.unlink()
        return jsonify({"deleted": name})
    return jsonify({"error": "not found"}), 404


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """
    Body: { pages: [{name, corners}], filename: "scan.pdf" }
    Saves the PDF to output/ and returns it as a download.
    """
    data     = request.get_json()
    pages_in = data.get("pages", [])
    filename = data.get("filename", "scan.pdf")
    paper    = data.get("paper", "a4")
    mode     = data.get("mode", "bw")

    processed = []
    errors    = []
    for item in pages_in:
        try:
            page = process_page(INPUT_DIR / item["name"], item["corners"], dpi=300, paper=paper, mode=mode)
            processed.append(page)
        except Exception as e:
            errors.append(f"{item['name']}: {e}")

    if not processed:
        return jsonify({"error": "No pages processed", "details": errors}), 500

    output_path = OUTPUT_DIR / filename
    build_pdf(processed, output_path)

    return send_file(
        str(output_path.resolve()),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("\n  Scanner UI running → http://localhost:5001\n")
    app.run(debug=False, port=5001, threaded=True)
