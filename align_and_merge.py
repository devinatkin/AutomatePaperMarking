import argparse
import os
from PyPDF2 import PdfReader, PdfWriter
import fitz
from PIL import Image
import cv2
import numpy as np

def convert_pdf_to_images(pdf_path, save_dir, dpi=300):
    doc = fitz.open(pdf_path)
    images = []
    zoom = dpi / 72  # 72 DPI is the default; scale accordingly
    mat = fitz.Matrix(zoom, zoom)
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)  # alpha=False to get RGB
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        out_path = os.path.join(save_dir, f"page_{i + 1}.jpg")
        img.save(out_path, "JPEG", quality=95, optimize=True)
        images.append(img)
    doc.close()
    return images

def get_aruco_corners(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # Create DetectorParameters with a backwards-compatible fallback for different OpenCV versions
    try:
        parameters = cv2.aruco.DetectorParameters_create()
    except AttributeError:
        parameters = cv2.aruco.DetectorParameters()

    # Try the module-level detectMarkers first (older OpenCV-contrib versions),
    # otherwise fall back to the ArucoDetector class (newer OpenCV versions).
    try:
        res = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    except AttributeError:
        try:
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            res = detector.detectMarkers(gray)
        except AttributeError:
            # Neither API is available
            return None, None

    # detectMarkers / detectMarkers via ArucoDetector may return 2 or 3 items
    if isinstance(res, tuple) and len(res) == 3:
        corners, ids, _ = res
    elif isinstance(res, tuple) and len(res) == 2:
        corners, ids = res
    else:
        # As a final fallback, try to index result
        try:
            corners = res[0]
            ids = res[1]
        except Exception:
            return None, None

    if ids is not None and len(ids) >= 4:
        return corners, ids
    return None, None

def sort_pages(corner_image_list):
    if len(corner_image_list) < 2:
        print("Not enough images with detected ArUco markers to align.")
        return None, None

    images, all_corners, all_ids = zip(*corner_image_list)

    id_sets = [set(id.flatten()) for id in all_ids]
    unique_id_sets = set(frozenset(s) for s in id_sets)

    page_count = len(unique_id_sets)
    print(f"Detected {page_count} unique pages based on ArUco markers.")

    pages = [[] for _ in range(page_count)]
    for i in range(len(images)):
        for j, unique_ids in enumerate(unique_id_sets):
            if id_sets[i] == unique_ids:
                pages[j].append([images[i], all_corners[i], all_ids[i]])
                break

    return pages

def align_page_images(page_images):
    print(f"Aligning {len(page_images)} images for a single page...")

    images, corners, ids = zip(*page_images)
    images = list(images)

    ref_image = images[0]
    ref_corners = corners[0]
    ref_ids = ids[0]

    for i in range(1, len(images)):
        img = images[i]
        img_corners = corners[i]
        img_ids = ids[i]

        # Common IDs are guaranteed based on previous filtering
        common_ids = set(ref_ids.flatten()).intersection(set(img_ids.flatten()))

        print(f"Found {len(common_ids)} common ArUco markers for alignment.")

        ref_pts = []
        img_pts = []
        for cid in common_ids:
            ref_index = np.where(ref_ids == cid)[0][0]
            img_index = np.where(img_ids == cid)[0][0]
            ref_pts.append(ref_corners[ref_index][0])  # Top-left corner
            img_pts.append(img_corners[img_index][0])  # Top-left corner

        # Convert collected marker corner arrays into one 2D point per marker (centroid)
        try:
            ref_arr = np.array([np.array(r).reshape(-1, 2).mean(axis=0) for r in ref_pts])
            img_arr = np.array([np.array(r).reshape(-1, 2).mean(axis=0) for r in img_pts])
        except Exception:
            # Fallback if points are already simple 2D points
            ref_arr = np.array(ref_pts)
            img_arr = np.array(img_pts)

        # Need at least 4 corresponding points to compute a reliable homography
        if ref_arr.ndim != 2 or img_arr.ndim != 2 or ref_arr.shape[0] < 4 or img_arr.shape[0] < 4:
            print("Not enough valid marker correspondences (need >= 4) to compute homography; skipping this image.")
            continue

        # Ensure correct dtype
        src_pts = img_arr.astype(np.float32)
        dst_pts = ref_arr.astype(np.float32)

        try:
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        except cv2.error as e:
            print(f"cv2.findHomography failed: {e}")
            H = None

        if H is not None:
            # PIL image.size -> (width, height)
            w, h = ref_image.size
            aligned_img = cv2.warpPerspective(np.array(img), H, (w, h))
            aligned_pil_img = Image.fromarray(aligned_img)
            
            # Save over the original image for simplicity
            images[i] = aligned_pil_img
            print(f"Aligned image {i+1} to reference image.")

        else:
            print("Homography computation failed.")

    return images

def merge_aligned_page_images(aligned_images):
    print(f"Merging {len(aligned_images)} aligned images...")
    # Convert all images to numpy arrays
    np_images = [np.array(img) for img in aligned_images]
    # Stack images and compute the pixel-wise minimum
    merged_array = np.minimum.reduce(np_images)
    merged_image = Image.fromarray(merged_array)
    return merged_image

def convert_images_to_pdf(image_files, output_pdf):
    pdf_writer = PdfWriter()
    for image_file in image_files:
        img = Image.open(image_file)
        img_rgb = img.convert('RGB')
        temp_pdf_path = image_file.replace('.jpg', '.pdf')
        img_rgb.save(temp_pdf_path, "PDF", resolution=100.0)
        pdf_reader = PdfReader(temp_pdf_path)
        pdf_writer.add_page(pdf_reader.pages[0])
        os.remove(temp_pdf_path)  # Clean up temporary PDF file
    with open(output_pdf, 'wb') as out_f:
        pdf_writer.write(out_f)
    print(f"Saved merged PDF to {output_pdf}")

def main():
    print("Starting the alignment and merging process...")
    # Take in a pdf file and align the pages
    parser = argparse.ArgumentParser(description='Align and merge two groups of darkest pages from a PDF file.')
    parser.add_argument('input_pdf', type=str, help='Path to the input PDF file')
    parser.add_argument('output_pdf', type=str, help='Path to the output PDF file')

    args = parser.parse_args()

    input_pdf = args.input_pdf
    output_pdf = args.output_pdf

    if not os.path.isfile(input_pdf):
        print(f"Input file {input_pdf} does not exist.")
        return
    if not input_pdf.lower().endswith('.pdf'):
        print("Input file must be a PDF.")
        return
    

    # Create a temporary directory to store images
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    images = convert_pdf_to_images(input_pdf, temp_dir)
    print(f"Converted {len(images)} pages to images.")
    corner_image_list = []
    for image in images:
        corners, ids = get_aruco_corners(image)
        if corners is not None:
            corner_image_list.append((image, corners, ids))
        else:
            print("No ArUco markers detected.")

    sorted_pages = sort_pages(corner_image_list)
    aligned_pages = []
    for page in sorted_pages:
        aligned_page = align_page_images(page)
        aligned_pages.append(aligned_page)

    # Merge the aligned images together
    merged_image_files = []
    for page_images in aligned_pages:
        merged_image = merge_aligned_page_images(page_images)
        merged_image_path = os.path.join(temp_dir, f"merged_page_{len(merged_image_files) + 1}.jpg")
        merged_image.save(merged_image_path, "JPEG", quality=95, optimize=True)
        merged_image_files.append(merged_image_path)

    convert_images_to_pdf(merged_image_files, output_pdf)

    # Clean up temporary images
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)

if __name__ == "__main__":
    main()