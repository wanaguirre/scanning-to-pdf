# scanning-to-pdf

A local scanner app that converts phone photos into a clean black-and-white PDF — one page per image.

## Features

- **Web UI** — drag the 4 corner handles to correct perspective, preview the warp, generate the PDF
- **Upload & manage images** directly from the browser (drag-and-drop or file picker)
- **iPhone / HEIC support** out of the box
- **CLI mode** if you just want to batch-process without a browser
- Adaptive threshold for clean black-on-white document look
- A4 output at 300 dpi

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Web UI (recommended)
```bash
python app.py
# → open http://localhost:5001
```

1. Upload images via the sidebar (drag-and-drop or **+ Add** button)
2. Select each page — auto-detected corners appear as draggable handles
3. Drag TL/TR/BR/BL handles to correct the perspective if needed
4. Click **👁 Preview** to check the result
5. Click **⬇ Generate PDF** to download

### CLI
```bash
# Drop images into input/, then:
python scan.py                    # auto-detect corners, output/scan.pdf
python scan.py --no-crop          # skip perspective correction
python scan.py --dpi 150          # smaller file size
python scan.py --output doc.pdf   # custom filename
```

## Project structure

```
scanning-to-pdf/
├── app.py            # Flask web server
├── scan.py           # CLI
├── scanner.py        # Core image processing library
├── templates/
│   └── index.html    # Single-page web UI
├── input/            # Source images (ignored by git)
├── output/           # Generated PDFs
└── requirements.txt
```
