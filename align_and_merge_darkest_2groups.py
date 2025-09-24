#!/usr/bin/env python3
"""
Blind align sheets using ArUco markers, then split warped sheets into 2 groups by perceptual hash,
and merge each group by darkest pixel. No master PDF, no ID matching.

Now with optional final thresholding to harden text and suppress ghosting.

Usage:
  python align_and_merge_darkest_2groups.py sheet_scans.pdf merged_2page.pdf \
    --dpi 500 --aruco-dict 4X4_1000 --marker-cluster-eps 0.20 --min-markers-per-sheet 3 \
    --final-threshold otsu --morph-open 3 --debug-dir debug_out
"""

import argparse
import io
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import fitz  # PyMuPDF
from PIL import Image

import cv2
aruco = cv2.aruco

DICT_NAMES = {
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

@dataclass
class SheetQuad:
    tl: np.ndarray
    tr: np.ndarray
    br: np.ndarray
    bl: np.ndarray

# ---------- PDF I/O ----------

import numpy as np
import cv2

def quad_from_all_marker_corners(corners: list[np.ndarray]) -> SheetQuad | None:
    """
    Fit a sheet quadrilateral from ALL detected marker corners using
    convex hull + minimum area rectangle. Returns (tl,tr,br,bl) ordered for image coords.
    """
    if not corners:
        return None
    # Concatenate all 4x2 corner sets into Nx2 cloud
    pts = np.vstack([c.reshape(-1, 2) for c in corners]).astype(np.float32)

    # Convex hull to suppress outliers
    hull = cv2.convexHull(pts)

    # Min area rectangle over hull
    rect = cv2.minAreaRect(hull)        # ((cx,cy),(w,h),angle)
    box  = cv2.boxPoints(rect)          # 4x2 float32, unsorted
    box  = box.astype(np.float32)

    # Sort to tl,tr,br,bl in image coords (x right, y down)
    s = box[:,0] + box[:,1]
    d = box[:,0] - box[:,1]
    tl = box[np.argmin(s)]
    br = box[np.argmax(s)]
    tr = box[np.argmin(d)]
    bl = box[np.argmax(d)]
    ordered = np.stack([tl,tr,br,bl], axis=0)

    # Enforce clockwise order (negative area with y down)
    x = ordered[:,0]; y = ordered[:,1]
    area2 = (x[0]*y[1] - x[1]*y[0]) + (x[1]*y[2] - x[2]*y[1]) + \
            (x[2]*y[3] - x[3]*y[2]) + (x[3]*y[0] - x[0]*y[3])
    if area2 > 0:
        ordered[[1,3]] = ordered[[3,1]]
    return SheetQuad(tl=ordered[0], tr=ordered[1], br=ordered[2], bl=ordered[3])


def pdf_pages_to_images(pdf_path: str, dpi: int) -> List[np.ndarray]:
    scale = dpi / 72.0
    images = []
    with fitz.open(pdf_path) as doc:
        for p in range(len(doc)):
            page = doc.load_page(p)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            images.append(img.copy())
    return images

def image_to_pdf_page(img_gray: np.ndarray, doc: fitz.Document, dpi: int):
    H, W = img_gray.shape
    pil = Image.fromarray(img_gray)
    buf = io.BytesIO(); pil.save(buf, format="PNG")
    width_pts = (W / dpi) * 72.0; height_pts = (H / dpi) * 72.0
    page = doc.new_page(width=width_pts, height=height_pts)
    page.insert_image(fitz.Rect(0,0,width_pts,height_pts), stream=buf.getvalue())

# ---------- Detection / preprocessing ----------

def build_detector(dict_name: str) -> aruco.ArucoDetector:
    dictionary = aruco.getPredefinedDictionary(DICT_NAMES[dict_name])
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 35
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.minCornerDistanceRate = 0.02
    params.minDistanceToBorder = 1
    params.minOtsuStdDev = 3.0
    return aruco.ArucoDetector(dictionary, params)

def preprocess_variants(bgr: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = [gray]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    out.append(clahe.apply(gray))
    at_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 35, 5)
    out.append(at_mean)
    at_gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 35, 5)
    out.append(at_gauss)
    bl = cv2.bilateralFilter(gray, d=7, sigmaColor=25, sigmaSpace=25)
    out.append(bl)
    return out

def detect_aruco_any(det: aruco.ArucoDetector, bgr: np.ndarray):
    for v in preprocess_variants(bgr):
        corners, ids, _ = det.detectMarkers(v)
        if ids is not None and len(ids) > 0:
            return corners, ids
    return [], np.empty((0,1), dtype=np.int32)

def centers_from_corners(corners: List[np.ndarray]) -> np.ndarray:
    if not corners:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([c.reshape(-1, 2).mean(axis=0) for c in corners], dtype=np.float32)

# ---------- Clustering sheets per scan page ----------

def cluster_points(points: np.ndarray, eps: float) -> List[List[int]]:
    N = len(points)
    visited = np.zeros(N, dtype=bool)
    clusters = []
    for i in range(N):
        if visited[i]: continue
        cluster = []
        stack = [i]; visited[i] = True
        while stack:
            j = stack.pop(); cluster.append(j)
            d2 = np.sum((points - points[j])**2, axis=1)
            neigh = np.where(d2 <= eps*eps)[0]
            for k in neigh:
                if not visited[k]:
                    visited[k] = True
                    stack.append(k)
        clusters.append(cluster)
    return clusters

def farthest_four(pts: np.ndarray) -> np.ndarray:
    M = len(pts)
    if M <= 4:
        return pts.copy()
    D = np.linalg.norm(pts[:,None,:] - pts[None,:,:], axis=2)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    chosen = [i, j]
    while len(chosen) < 4:
        mask = np.ones(M, dtype=bool); mask[chosen] = False
        if not mask.any(): break
        dist_to_set = np.min(D[mask][:, chosen], axis=1)
        k_candidates = np.where(mask)[0]
        k = k_candidates[np.argmax(dist_to_set)]
        chosen.append(int(k))
    return pts[np.array(chosen)]

def order_quad(pts: np.ndarray) -> SheetQuad:
    """
    Return corners as (tl, tr, br, bl) in image coords (x right, y down).
    Uses sum and (x - y) difference; then enforces clockwise orientation.
    """
    s = pts[:, 0] + pts[:, 1]        # x + y
    d = pts[:, 0] - pts[:, 1]        # x - y

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    ordered = np.stack([tl, tr, br, bl], axis=0)

    # Enforce clockwise order (shoelace area negative in image coords with y down)
    x = ordered[:, 0]; y = ordered[:, 1]
    area2 = (x[0]*y[1] - x[1]*y[0]) + (x[1]*y[2] - x[2]*y[1]) + \
            (x[2]*y[3] - x[3]*y[2]) + (x[3]*y[0] - x[0]*y[3])
    if area2 > 0:
        ordered[[1, 3]] = ordered[[3, 1]]

    return SheetQuad(tl=ordered[0], tr=ordered[1], br=ordered[2], bl=ordered[3])

# ---------- Warping and merging ----------

def estimate_sheet_size(quad: SheetQuad) -> Tuple[int, int]:
    def dist(a, b): return float(np.linalg.norm(a - b))
    W = int(round(0.5 * (dist(quad.tl, quad.tr) + dist(quad.bl, quad.br))))
    H = int(round(0.5 * (dist(quad.tl, quad.bl) + dist(quad.tr, quad.br))))
    return max(W,256), max(H,256)

def homography_warp(img: np.ndarray, quad: SheetQuad, out_wh: Tuple[int, int]) -> np.ndarray:
    W, H = out_wh
    src = np.float32([quad.tl, quad.tr, quad.br, quad.bl])
    # Map so that top-left is (0,0), y increases downward; fix vertical flip
    dst = np.float32([[0, H-1], [W-1, H-1], [W-1, 0], [0, 0]])
    Hmat = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, Hmat, (W, H), flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

def merge_darkest_gray(images_bgr: List[np.ndarray]) -> np.ndarray:
    if not images_bgr:
        raise RuntimeError("No warped sheets to merge.")
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images_bgr]
    # light normalization so faint pencil marks win the min
    grays = [cv2.equalizeHist(g) for g in grays]
    merged = grays[0].copy()
    for g in grays[1:]:
        np.minimum(merged, g, out=merged)
    return merged

# ---------- Final thresholding & morphology (NEW) ----------

def apply_final_threshold(img_gray: np.ndarray,
                          method: str,
                          percentile: float = 75.0,
                          adaptive_block: int = 35,
                          adaptive_C: int = 5) -> np.ndarray:
    """
    Returns a uint8 0/255 image. method in {"none","otsu","percentile","adaptive"}.
    """
    if method == "none":
        return img_gray

    if method == "otsu":
        _t, bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw

    if method == "percentile":
        # clamp percentile and compute robust global threshold
        p = float(np.clip(percentile, 0.0, 100.0))
        thr = np.percentile(img_gray, p)
        _t, bw = cv2.threshold(img_gray, int(thr), 255, cv2.THRESH_BINARY)
        return bw

    if method == "adaptive":
        blk = adaptive_block if adaptive_block % 2 == 1 else adaptive_block + 1
        bw = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, blk, adaptive_C)
        return bw

    # Fallback: no change if unknown method
    return img_gray

def apply_morphology(img_gray: np.ndarray, open_k: int = 0, close_k: int = 0) -> np.ndarray:
    """
    Applies opening/closing on a binary image. Kernel sizes 0 mean 'skip'.
    """
    out = img_gray
    if open_k and open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_k and close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out

# ---------- Perceptual hash + K-means (K=2) ----------

def phash_gray(img_gray: np.ndarray, hash_size: int = 16, highfreq_factor: int = 4) -> np.ndarray:
    size = hash_size * highfreq_factor
    small = cv2.resize(img_gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low)
    bits = dct_low > med
    return bits.astype(np.uint8).flatten()

def kmeans_binary_hashes(hashes: List[np.ndarray], k: int = 2, iters: int = 25, seed: int = 1337) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H = np.stack(hashes)  # N x D
    N, D = H.shape
    idx = rng.choice(N, size=min(k, N), replace=False)
    C = H[idx].astype(np.float32)  # centroids in [0..1]
    for _ in range(iters):
        # assign using Hamming via thresholded centroids
        dists = np.zeros((N, C.shape[0]), dtype=np.int32)
        for j in range(C.shape[0]):
            pred = (C[j] >= 0.5).astype(np.uint8)
            dists[:, j] = np.count_nonzero(H != pred, axis=1)
        labels = np.argmin(dists, axis=1)
        newC = np.zeros_like(C)
        for j in range(C.shape[0]):
            members = H[labels == j]
            if len(members) == 0:
                newC[j] = H[rng.integers(0, N)]
            else:
                newC[j] = members.mean(axis=0)
        if np.allclose(newC, C):
            break
        C = newC
    return labels

# ---------- Main pipeline ----------

def extract_sheet_quads(img: np.ndarray, det: aruco.ArucoDetector,
                        cluster_eps_frac: float, min_markers_per_sheet: int) -> List[SheetQuad]:
    corners, ids = detect_aruco_any(det, img)
    if ids.size == 0:
        return []
    # cluster by marker centres (okay for grouping)
    ctrs = centers_from_corners(corners)
    h, w, _ = img.shape
    eps = cluster_eps_frac * min(w, h)
    clusters = cluster_points(ctrs, eps)

    quads: List[SheetQuad] = []
    for cl in clusters:
        if len(cl) < min_markers_per_sheet:
            continue
        # pull the actual corner arrays for members of this cluster
        cluster_corners = [corners[i] for i in cl]
        q = quad_from_all_marker_corners(cluster_corners)
        if q is not None:
            quads.append(q)
    return quads


def main():
    ap = argparse.ArgumentParser(description="Blind ArUco alignment into 2 groups, darkest-merge to a 2-page PDF.")
    ap.add_argument("input_pdf", help="PDF with scans; may contain multiple sheets per page.")
    ap.add_argument("output_pdf", help="Output PDF with 1 page per cluster (default K=2).")
    ap.add_argument("--dpi", type=int, default=500, help="Render DPI (higher is safer for 8 mm tags).")
    ap.add_argument("--aruco-dict", default="4X4_1000", choices=sorted(DICT_NAMES.keys()))
    ap.add_argument("--marker-cluster-eps", type=float, default=0.20,
                    help="Clustering radius as fraction of min(image dimension).")
    ap.add_argument("--min-markers-per-sheet", type=int, default=3,
                    help="Require at least this many markers to accept a sheet.")
    ap.add_argument("--override-size-mm", type=float, nargs=2, metavar=("W_MM", "H_MM"),
                    help="Force output sheet size (e.g., 215.9 279.4 for Letter).")
    ap.add_argument("--debug-dir", default=None, help="Optional folder for debug warped sheets.")

    # New thresholding options
    ap.add_argument("--final-threshold", default="none",
                    choices=["none", "otsu", "percentile", "adaptive"],
                    help="Final binarisation method applied after merging.")
    ap.add_argument("--percentile", type=float, default=75.0,
                    help="Percentile for 'percentile' thresholding (0..100).")
    ap.add_argument("--adaptive-block", type=int, default=35,
                    help="Odd window size for 'adaptive' thresholding.")
    ap.add_argument("--adaptive-C", type=int, default=5,
                    help="Constant subtracted in 'adaptive' thresholding.")

    ap.add_argument("--morph-open", type=int, default=0,
                    help="Apply morphological opening with NxN kernel (N>0).")
    ap.add_argument("--morph-close", type=int, default=0,
                    help="Apply morphological closing with NxN kernel (N>0).")

    args = ap.parse_args()

    det = build_detector(args.aruco_dict)
    pages = pdf_pages_to_images(args.input_pdf, dpi=args.dpi)

    warped: List[np.ndarray] = []
    ref_size: Tuple[int,int] | None = None

    for page_idx, img in enumerate(pages):
        quads = extract_sheet_quads(img, det, args.marker_cluster_eps, args.min_markers_per_sheet)
        if not quads:
            print(f"[WARN] No sheets found on page {page_idx+1}.")
            continue
        for qi, q in enumerate(quads):
            if args.override_size_mm:
                w_mm, h_mm = args.override_size_mm
                W = int(round((w_mm/25.4)*args.dpi))
                H = int(round((h_mm/25.4)*args.dpi))
                out_wh = (W, H)
            elif ref_size is None:
                ref_size = estimate_sheet_size(q)
                out_wh = ref_size
            else:
                out_wh = ref_size
            w = homography_warp(img, q, out_wh)
            warped.append(w)
            if args.debug_dir:
                os.makedirs(args.debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(args.debug_dir, f"warped_p{page_idx+1:03d}_s{qi+1:02d}.png"), w)

    if not warped:
        raise SystemExit("No sheets detected in any page. Increase --dpi or --marker-cluster-eps.")

    # compute perceptual hashes and labels (K=2 by construction)
    hashes = []
    for w in warped:
        g = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)
        g = cv2.equalizeHist(g)
        hashes.append(phash_gray(g, hash_size=16, highfreq_factor=4))

    labels = kmeans_binary_hashes(hashes, k=2)
    groups: Dict[int, List[np.ndarray]] = {}
    for lab, w in zip(labels, warped):
        groups.setdefault(int(lab), []).append(w)

    # order groups by size (largest first) so page 1 is the most common template
    ordered = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)

    doc = fitz.open()
    for gi, (lab, imgs) in enumerate(ordered, start=1):
        merged_gray = merge_darkest_gray(imgs)

        # Apply final thresholding and optional morphology (NEW)
        bw = apply_final_threshold(
            merged_gray,
            method=args.final_threshold,
            percentile=args.percentile,
            adaptive_block=args.adaptive_block,
            adaptive_C=args.adaptive_C,
        )
        if args.final_threshold != "none":
            bw = apply_morphology(bw, open_k=args.morph_open, close_k=args.morph_close)
            image_to_pdf_page(bw, doc, dpi=args.dpi)
        else:
            image_to_pdf_page(merged_gray, doc, dpi=args.dpi)

        print(f"[OK] Cluster {gi} (label {lab}): merged {len(imgs)} sheet(s). "
              f"threshold={args.final_threshold}")

    doc.save(args.output_pdf); doc.close()
    print(f"[DONE] Wrote {len(ordered)} page(s) to {args.output_pdf}")

if __name__ == "__main__":
    main()
