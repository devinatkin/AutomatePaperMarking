#!/usr/bin/env python3
"""
stamp_aruco_corners.py
Add ArUco fiducial markers to the four corners of each page in a PDF.

Dependencies:
  - opencv-python-headless
  - PyMuPDF (fitz)
  - numpy

Example:
  python stamp_aruco_corners.py input.pdf output.pdf \
    --marker-mm 8 --inset-mm 6 --dict 4X4_1000 --unique-ids
"""

import argparse
import io
import sys
import math
import numpy as np
import fitz  # PyMuPDF

# OpenCV's ArUco lives under cv2.aruco; on some builds it's a contrib module.
try:
    import cv2
    aruco = cv2.aruco
except Exception as e:
    print("Error: OpenCV ArUco module not available. Install opencv-python-headless.", file=sys.stderr)
    raise

# -------- Helpers --------

ARUCO_DICT_NAMES = {
    # name -> cv2 constant
    "4X4_50": aruco.DICT_4X4_50,
    "4X4_100": aruco.DICT_4X4_100,
    "4X4_250": aruco.DICT_4X4_250,
    "4X4_1000": aruco.DICT_4X4_1000,
    "5X5_50": aruco.DICT_5X5_50,
    "5X5_100": aruco.DICT_5X5_100,
    "5X5_250": aruco.DICT_5X5_250,
    "5X5_1000": aruco.DICT_5X5_1000,
    "6X6_50": aruco.DICT_6X6_50,
    "6X6_100": aruco.DICT_6X6_100,
    "6X6_250": aruco.DICT_6X6_250,
    "6X6_1000": aruco.DICT_6X6_1000,
    "7X7_50": aruco.DICT_7X7_50,
    "7X7_100": aruco.DICT_7X7_100,
    "7X7_250": aruco.DICT_7X7_250,
    "7X7_1000": aruco.DICT_7X7_1000,
    "APRILTAG_16h5": aruco.DICT_APRILTAG_16h5,
    "APRILTAG_25h9": aruco.DICT_APRILTAG_25h9,
    "APRILTAG_36h10": aruco.DICT_APRILTAG_36h10,
    "APRILTAG_36h11": aruco.DICT_APRILTAG_36h11,
}

def mm_to_pts(mm: float) -> float:
    # 1 inch = 25.4 mm, PDF units are points at 72 dpi
    return (mm / 25.4) * 72.0

def make_aruco_marker_png_bytes(dictionary_name: str, marker_id: int, px: int, border_bits: int = 1) -> bytes:
    if dictionary_name not in ARUCO_DICT_NAMES:
        raise ValueError(f"Unknown dictionary '{dictionary_name}'.")
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT_NAMES[dictionary_name])
    img = aruco.generateImageMarker(dictionary, marker_id, px, borderBits=border_bits)
    # Ensure 8-bit single channel to PNG
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    success, buf = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode marker PNG.")
    return buf.tobytes()

def corner_rects(page_rect, marker_pts, inset_pts):
    """
    Compute rectangles (fitz.Rect) for TL, TR, BR, BL given page_rect,
    a marker square size in points, and inset from edges in points.
    """
    w = marker_pts
    h = marker_pts
    x0, y0, x1, y1 = page_rect  # left, top, right, bottom in points

    # PyMuPDF coords: origin at top-left, y increases downward.
    # TL
    tl = fitz.Rect(x0 + inset_pts, y0 + inset_pts, x0 + inset_pts + w, y0 + inset_pts + h)
    # TR
    tr = fitz.Rect(x1 - inset_pts - w, y0 + inset_pts, x1 - inset_pts, y0 + inset_pts + h)
    # BR
    br = fitz.Rect(x1 - inset_pts - w, y1 - inset_pts - h, x1 - inset_pts, y1 - inset_pts)
    # BL
    bl = fitz.Rect(x0 + inset_pts, y1 - inset_pts - h, x0 + inset_pts + w, y1 - inset_pts)
    return [tl, tr, br, bl]

def gen_ids_for_page(page_index: int, reuse_ids: bool, dict_capacity: int = 1000):
    """
    Return four IDs [TL, TR, BR, BL].
    If reuse_ids is False, id = page_index*4 + corner_index.
    Raises if exceeding dictionary capacity.
    """
    if reuse_ids:
        return [0, 1, 2, 3]
    base = page_index * 4
    ids = [base + i for i in range(4)]
    if any(i >= dict_capacity for i in ids):
        raise ValueError(
            f"Marker ID {max(ids)} exceeds dictionary capacity {dict_capacity}. "
            "Use a larger dictionary or enable --reuse-ids."
        )
    return ids

def dict_capacity_from_name(name: str) -> int:
    # Rough maximum IDs per dictionary family.
    # OpenCV docs specify: 4x4_1000 -> 1000, 5x5_1000 -> 1000, etc.
    if name.endswith("_50"): return 50
    if name.endswith("_100"): return 100
    if name.endswith("_250"): return 250
    if name.endswith("_1000"): return 1000
    # AprilTag families have fixed sets; treat conservatively.
    if name.startswith("APRILTAG"):
        return 6000  # generous upper bound for safety
    return 250  # fallback

# -------- Main --------

def process_pdf(
    input_pdf: str,
    output_pdf: str,
    marker_mm: float,
    inset_mm: float,
    dictionary_name: str,
    per_page_unique_ids: bool,
    password: str | None,
    marker_render_px: int,
    border_bits: int,
    opacity: float,
):
    # Convert sizes to points
    marker_pts = mm_to_pts(marker_mm)
    inset_pts = mm_to_pts(inset_mm)

    cap = dict_capacity_from_name(dictionary_name)

    # Pre-generate a cache of images by id -> png bytes to avoid recompute
    marker_png_cache: dict[int, bytes] = {}

    # Open input
    doc = fitz.open(input_pdf)
    if doc.needs_pass:
        if not password:
            raise RuntimeError("PDF is encrypted. Provide --password.")
        if not doc.authenticate(password):
            raise RuntimeError("Failed to authenticate with provided password.")

    # New output document
    out = fitz.open()

    for pidx in range(len(doc)):
        page = doc.load_page(pidx)
        # Work on a new page (copy to preserve content + allow stamping)
        outpage = out.new_page(width=page.rect.width, height=page.rect.height)
        outpage.show_pdf_page(outpage.rect, doc, pidx)

        ids = gen_ids_for_page(pidx, reuse_ids=not per_page_unique_ids, dict_capacity=cap)
        rects = corner_rects(outpage.rect, marker_pts, inset_pts)

        # TL, TR, BR, BL order
        for corner_idx, rect in enumerate(rects):
            marker_id = ids[corner_idx]
            if marker_id not in marker_png_cache:
                marker_png_cache[marker_id] = make_aruco_marker_png_bytes(
                    dictionary_name, marker_id, marker_render_px, border_bits=border_bits
                )

            # Insert image scaled to rect
            # Use a transparency-aware method: open as image stream each time.
            png_bytes = marker_png_cache[marker_id]
            # PyMuPDF insert can apply blend via 'opacity'
            outpage.insert_image(
                rect,
                stream=png_bytes,
                keep_proportion=False,  # force exact square in points
                overlay=True
            )

    out.save(output_pdf)
    out.close()
    doc.close()

def main():
    parser = argparse.ArgumentParser(description="Stamp ArUco markers into the four corners of each PDF page.")
    parser.add_argument("input_pdf", help="Path to input PDF")
    parser.add_argument("output_pdf", help="Path to output PDF")
    parser.add_argument("--marker-mm", type=float, default=8.0,
                        help="Marker square side length in millimetres (default: 8.0)")
    parser.add_argument("--inset-mm", type=float, default=6.0,
                        help="Inset from each page edge in millimetres (default: 6.0)")
    parser.add_argument("--dict", dest="dictionary_name", default="4X4_1000",
                        choices=sorted(ARUCO_DICT_NAMES.keys()),
                        help="ArUco/AprilTag dictionary (default: 4X4_1000)")
    parser.add_argument("--unique-ids", dest="unique_ids", action="store_true",
                        help="Use unique IDs per page and corner (page_index*4 + corner).")
    parser.add_argument("--reuse-ids", dest="unique_ids", action="store_false",
                        help="Reuse the same 4 IDs on every page (0..3).")
    parser.set_defaults(unique_ids=True)
    parser.add_argument("--password", default=None, help="PDF password if encrypted")
    parser.add_argument("--render-px", type=int, default=300,
                        help="Internal PNG render size in pixels (default: 300). Larger = crisper.")
    parser.add_argument("--border-bits", type=int, default=1,
                        help="ArUco borderBits around the marker (default: 1).")
    parser.add_argument("--opacity", type=float, default=1.0,
                        help="Image opacity from 0..1 (default 1.0).")

    args = parser.parse_args()

    if args.marker_mm <= 0 or args.inset_mm < 0:
        print("marker-mm must be > 0 and inset-mm must be >= 0", file=sys.stderr)
        sys.exit(2)
    if not (0.0 < args.opacity <= 1.0):
        print("opacity must be in (0,1]", file=sys.stderr)
        sys.exit(2)
    if args.render_px < 64:
        print("render-px should be at least 64 for legibility.", file=sys.stderr)
        sys.exit(2)

    try:
        process_pdf(
            input_pdf=args.input_pdf,
            output_pdf=args.output_pdf,
            marker_mm=args.marker_mm,
            inset_mm=args.inset_mm,
            dictionary_name=args.dictionary_name,
            per_page_unique_ids=args.unique_ids,
            password=args.password,
            marker_render_px=args.render_px,
            border_bits=args.border_bits,
            opacity=args.opacity,
        )
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
