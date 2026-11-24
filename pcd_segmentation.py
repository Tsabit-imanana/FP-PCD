"""
PCD Segmentation Library
Implementasi Komprehensif Metode Segmentasi Pengolahan Citra Digital (PCD)
Berdasarkan Analisis PCD_FINAL.ipynb (Kelompok 9 - PCD UPN "Veteran" Jatim)

Metode yang diimplementasikan:
1. Otsu Thresholding & Background Removal (Optimasi Morfologi & Flood Fill)
2. Edge-Based Segmentation (Canny Edge Detection - 5 Tahapan Teori)
3. Region-Based Segmentation (Region Growing BFS - 5 Tahapan Teori)
4. Clustering-Based Segmentation (K-Means Clustering - 7 Tahapan Teori)
5. Watershed Segmentation (Marker-Based - 8 Tahapan Teori)
6. Graph Cut Segmentation (GrabCut Algorithm - 8 Tahapan Teori)
"""

import cv2
import numpy as np
from collections import deque
from typing import Tuple, Dict, Any, Optional, List


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Memastikan citra berformat RGB 3-channel (uint8)."""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Memastikan citra berformat BGR 3-channel untuk operasi OpenCV."""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img


def create_transparent_image(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Menggabungkan citra RGB dengan binary mask ke format RGBA transparan."""
    rgba = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = mask
    return rgba


# ==============================================================================
# 1. OTSU THRESHOLDING & BACKGROUND REMOVAL (OPTIMASI MORFOLOGI & FLOOD FILL)
# ==============================================================================

def proses_otsu_thresholding(
    img_bgr: np.ndarray,
    blur_ksize: int = 5,
    morph_ksize: int = 5,
    morph_iters: int = 2,
    auto_invert: bool = True,
    keep_largest_component: bool = True,
    fill_holes: bool = True,
    smooth_mask: bool = True
) -> Dict[str, Any]:
    """
    Eksekusi pipeline Otsu Thresholding & Background Removal:
    1. Grayscale
    2. Gaussian Blur Noise Reduction
    3. Otsu Thresholding
    4. Border Background Inversion Detection
    5. Morphological Open & Close
    6. Connected Components (Largest White Area)
    7. Flood Fill Hole Filling
    8. Gaussian Edge Smoothing
    9. Alpha Transparency Generation
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tinggi, lebar = img_bgr.shape[:2]

    # Preprocessing
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blur = cv2.GaussianBlur(gray, (k, k), 0)

    # Otsu Thresholding
    threshold_value, raw_thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = raw_thresh.copy()

    # Periksa warna mayoritas pada batas (tepi) gambar
    tepi = np.concatenate([
        mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]
    ])
    inverted = False
    if auto_invert and np.mean(tepi) > 127:
        mask = cv2.bitwise_not(mask)
        inverted = True

    # Morfologi Opening & Closing
    kernel_m = np.ones((morph_ksize, morph_ksize), np.uint8)
    mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_m, iterations=morph_iters)
    mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel_m, iterations=morph_iters)

    # Connected Components (Pertahankan area terbesar)
    mask_bersih = mask_morph.copy()
    if keep_largest_component:
        jumlah, label, statistik, centroid = cv2.connectedComponentsWithStats(
            mask_morph, connectivity=8
        )
        if jumlah > 1:
            area_terbesar_idx = 1 + np.argmax(statistik[1:, cv2.CC_STAT_AREA])
            mask_bersih = np.zeros_like(mask_morph)
            mask_bersih[label == area_terbesar_idx] = 255

    # Hole Filling dengan Flood Fill
    mask_filled = mask_bersih.copy()
    if fill_holes:
        flood = mask_bersih.copy()
        flood_mask = np.zeros((tinggi + 2, lebar + 2), np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        lubang = cv2.bitwise_not(flood)
        mask_filled = cv2.bitwise_or(mask_bersih, lubang)

    # Haluskan tepi mask
    mask_final = mask_filled.copy()
    if smooth_mask:
        mask_final = cv2.GaussianBlur(mask_final, (5, 5), 0)

    # Citra Transparan BGRA & RGBA
    hasil_transparan_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    hasil_transparan_bgra[:, :, 3] = mask_final
    hasil_transparan_rgba = cv2.cvtColor(hasil_transparan_bgra, cv2.COLOR_BGRA2RGBA)

    # Citra Foreground pada kanvas putih
    fg_white = np.ones_like(img_rgb) * 255
    alpha = (mask_final.astype(float) / 255.0)[:, :, np.newaxis]
    fg_white = (img_rgb * alpha + fg_white * (1.0 - alpha)).astype(np.uint8)

    return {
        "img_rgb": img_rgb,
        "gray": gray,
        "blur": blur,
        "threshold_value": float(threshold_value),
        "raw_thresh": raw_thresh,
        "inverted": inverted,
        "mask_morph": mask_morph,
        "mask_bersih": mask_bersih,
        "mask_final": mask_final,
        "transparent_rgba": hasil_transparan_rgba,
        "foreground_white": fg_white
    }


# ==============================================================================
# 2. EDGE-BASED SEGMENTATION (CANNY EDGE DETECTION - 5 TAHAPAN)
# ==============================================================================

def proses_canny_edge(
    img_bgr: np.ndarray,
    blur_ksize: int = 5,
    blur_sigma: float = 1.4,
    auto_threshold: bool = True,
    low_thresh: int = 50,
    high_thresh: int = 150,
    sigma_ratio: float = 0.33
) -> Dict[str, Any]:
    """
    Eksekusi 5 Tahapan Teori Edge-Based Segmentation:
    Tahap 1: Grayscale (Konversi ke citra keabuan)
    Tahap 2: Gaussian Blur (Pengurangan noise dengan kernel & sigma)
    Tahap 3: Deteksi Gradien (Operator Sobel X & Y, Magnitude & Normalisasi)
    Tahap 4 & 5: Non-Maximum Suppression (NMS) & Hysteresis Thresholding (Canny Edge)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Tahap 1: Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Tahap 2: Gaussian Blur
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blur = cv2.GaussianBlur(gray, (k, k), blur_sigma)

    # Tahap 3: Deteksi Gradien Sobel
    grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    gradient_magnitude_norm = cv2.normalize(
        gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Tahap 4 & 5: NMS & Hysteresis Thresholding (Canny Edge)
    if auto_threshold:
        v = np.median(blur)
        threshold_bawah = int(max(0, (1.0 - sigma_ratio) * v))
        threshold_atas = int(min(255, (1.0 + sigma_ratio) * v))
        if threshold_atas - threshold_bawah < 40:
            threshold_bawah = 50
            threshold_atas = 150
    else:
        threshold_bawah = low_thresh
        threshold_atas = high_thresh

    edges = cv2.Canny(blur, threshold_bawah, threshold_atas)

    # Overlay tepi pada citra asli (merah menyala)
    overlay = img_rgb.copy()
    overlay[edges > 0] = [255, 0, 0]

    return {
        "img_rgb": img_rgb,
        "gray": gray,
        "blur": blur,
        "grad_x": np.uint8(np.absolute(grad_x)),
        "grad_y": np.uint8(np.absolute(grad_y)),
        "gradient_magnitude": gradient_magnitude_norm,
        "threshold_bawah": threshold_bawah,
        "threshold_atas": threshold_atas,
        "edges": edges,
        "overlay": overlay
    }


# ==============================================================================
# 3. REGION-BASED SEGMENTATION (REGION GROWING - 5 TAHAPAN)
# ==============================================================================

def proses_region_growing(
    img_bgr: np.ndarray,
    seed_point: Optional[Tuple[int, int]] = None,
    threshold: int = 30,
    connectivity: int = 8,
    blur_ksize: int = 5
) -> Dict[str, Any]:
    """
    Eksekusi 5 Tahapan Teori Region Growing:
    Tahap 1: Karakteristik Piksel (Grayscale & Pengurangan Noise)
    Tahap 2: Menentukan Titik Awal (Seed Point)
    Tahap 3 & 4: Memeriksa Tetangga & Menggabungkan Piksel Serupa (BFS Queue)
    Tahap 5: Menghasilkan Segmentasi Wilayah (Masking & Segmented Overlay)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tinggi, lebar = img_bgr.shape[:2]

    # Tahap 1: Grayscale & Blur
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blur = cv2.GaussianBlur(gray, (k, k), 0)

    # Tahap 2: Menentukan Titik Awal (Seed Point)
    if seed_point is None:
        seed_point = (lebar // 2, tinggi // 2)
    else:
        seed_point = (
            max(0, min(lebar - 1, int(seed_point[0]))),
            max(0, min(tinggi - 1, int(seed_point[1])))
        )

    # Visualisasi Seed Point pada citra asli (lingkaran merah berpusat di seed)
    img_seed_viz = img_rgb.copy()
    cv2.circle(img_seed_viz, seed_point, max(5, int(min(lebar, tinggi) * 0.02)), (255, 0, 0), -1)
    cv2.circle(img_seed_viz, seed_point, max(7, int(min(lebar, tinggi) * 0.025)), (255, 255, 255), 2)

    # Tahap 3 & 4: BFS Region Growing
    segmented = np.zeros((tinggi, lebar), dtype=np.uint8)
    visited = np.zeros((tinggi, lebar), dtype=bool)

    queue = deque([seed_point])
    visited[seed_point[1], seed_point[0]] = True
    seed_value = int(blur[seed_point[1], seed_point[0]])

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

    pixel_count = 0
    while queue:
        cx, cy = queue.popleft()
        segmented[cy, cx] = 255
        pixel_count += 1

        for dx, dy in neighbors:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < lebar and 0 <= ny < tinggi and not visited[ny, nx]:
                pixel_val = int(blur[ny, nx])
                if abs(pixel_val - seed_value) <= threshold:
                    visited[ny, nx] = True
                    queue.append((nx, ny))

    # Tahap 5: Masking & Segmented Image Overlay
    hasil_segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=segmented)
    overlay = img_rgb.copy()
    overlay[segmented == 255] = [0, 200, 255]
    blended = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)

    return {
        "img_rgb": img_rgb,
        "gray": gray,
        "blur": blur,
        "seed_point": seed_point,
        "seed_value": seed_value,
        "img_seed_viz": img_seed_viz,
        "mask_region": segmented,
        "hasil_segmented": hasil_segmented,
        "blended": blended,
        "pixel_count": pixel_count,
        "area_percentage": round((pixel_count / (tinggi * lebar)) * 100, 2)
    }


# ==============================================================================
# 4. CLUSTERING-BASED SEGMENTATION (K-MEANS CLUSTERING - 7 TAHAPAN)
# ==============================================================================

def proses_kmeans_clustering(
    img_rgb: np.ndarray,
    K: int = 3,
    max_iter: int = 100,
    epsilon: float = 1.0,
    sample_size: int = 100000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Eksekusi 7 Tahapan Teori K-Means Clustering:
    Tahap 1: Menentukan Jumlah Cluster (K)
    Tahap 2: Menentukan Centroid Awal (K-Means++ Initialization)
    Tahap 3: Menghitung Jarak Piksel (Euclidean Distance)
    Tahap 4: Mengelompokkan Piksel (Nearest Centroid Assignment)
    Tahap 5: Memperbarui Centroid (Center Re-calculation)
    Tahap 6: Pengulangan Proses Konvergensi
    Tahap 7: Menghasilkan Citra Segmentasi & Distribusi Cluster
    """
    tinggi, lebar, _ = img_rgb.shape
    pixels = img_rgb.reshape((-1, 3)).astype(np.float32)

    # Inisialisasi Centroid Awal K-Means++
    np.random.seed(random_state)
    n_samples = min(len(pixels), sample_size)
    sample_indices = np.random.choice(len(pixels), n_samples, replace=False)
    sample_pixels = pixels[sample_indices]

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max_iter,
        epsilon
    )

    _, _, centers_awal = cv2.kmeans(
        sample_pixels, K, None, criteria, 1, cv2.KMEANS_PP_CENTERS
    )

    # Perhitungan jarak awal Euclidean
    jarak_awal = np.linalg.norm(
        pixels[:, np.newaxis, :] - centers_awal[np.newaxis, :, :],
        axis=2
    )
    labels_pertama = np.argmin(jarak_awal, axis=1).astype(np.int32)
    label_pertama_visual = (
        labels_pertama.reshape((tinggi, lebar)) * (255 // max(K - 1, 1))
    ).astype(np.uint8)

    # Eksekusi Penuh K-Means Konvergensi
    compactness, labels, centers = cv2.kmeans(
        pixels, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()

    # Rekonstruksi Citra Hasil Segmentasi
    segmented = centers[labels].reshape((tinggi, lebar, 3))
    segmented = np.uint8(np.clip(segmented, 0, 255))

    # Visualisasi Label Cluster
    label_image = labels.reshape((tinggi, lebar))
    label_visual = (label_image * (255 // max(K - 1, 1))).astype(np.uint8)

    # Hitung distribusi jumlah piksel per cluster
    jumlah_piksel = np.bincount(labels, minlength=K)
    persentase_piksel = [(cnt / len(pixels)) * 100 for cnt in jumlah_piksel]

    # Palet warna visualizer (strip bar horizontal)
    palette_awal = np.zeros((60, 300, 3), dtype=np.uint8)
    for i, c in enumerate(centers_awal):
        x1 = int(i * 300 / K)
        x2 = int((i + 1) * 300 / K)
        palette_awal[:, x1:x2] = np.uint8(np.clip(c, 0, 255))

    palette_akhir = np.zeros((60, 300, 3), dtype=np.uint8)
    for i, c in enumerate(centers):
        x1 = int(i * 300 / K)
        x2 = int((i + 1) * 300 / K)
        palette_akhir[:, x1:x2] = np.uint8(np.clip(c, 0, 255))

    return {
        "img_rgb": img_rgb,
        "K": K,
        "centers_awal": centers_awal,
        "centers_akhir": centers,
        "palette_awal": palette_awal,
        "palette_akhir": palette_akhir,
        "label_pertama_visual": label_pertama_visual,
        "label_visual": label_visual,
        "label_raw": label_image,
        "segmented": segmented,
        "jumlah_piksel": jumlah_piksel.tolist(),
        "persentase_piksel": [round(p, 2) for p in persentase_piksel],
        "compactness": float(compactness)
    }


# ==============================================================================
# 5. WATERSHED SEGMENTATION (8 TAHAPAN)
# ==============================================================================

def proses_watershed_segmentation(
    img_bgr: np.ndarray,
    fg_ratio: float = 0.4,
    blur_ksize: int = 5,
    morph_ksize: int = 3,
    dilate_iters: int = 3,
    open_iters: int = 2
) -> Dict[str, Any]:
    """
    Eksekusi 8 Tahapan Teori Watershed:
    Tahap 1: Grayscale & Preprocessing Blur
    Tahap 2: Thresholding Otsu Inverted
    Tahap 3: Operasi Morfologi Opening
    Tahap 4: Menentukan Sure Background & Sure Foreground (Distance Transform)
    Tahap 5: Menentukan Unknown Region
    Tahap 6: Menentukan Marker Unik
    Tahap 7: Proses Watershed OpenCV (Garis batas merah)
    Tahap 8: Hasil Segmentasi Label Pseudo-color & Boundary Overlay
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tinggi, lebar = img_bgr.shape[:2]

    # Tahap 1: Grayscale & Blur
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blur = cv2.GaussianBlur(gray, (k, k), 0)

    # Tahap 2: Thresholding Biner Inverted
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Tahap 3: Operasi Morfologi Opening
    kernel_m = np.ones((morph_ksize, morph_ksize), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_m, iterations=open_iters)

    # Tahap 4: Sure Background & Sure Foreground
    sure_bg = cv2.dilate(opening, kernel_m, iterations=dilate_iters)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sure_fg = cv2.threshold(dist_transform, fg_ratio * dist_transform.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    # Tahap 5: Unknown Region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Tahap 6: Marker
    jumlah_objek, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Visualisasi Marker Berwarna
    marker_visual = np.zeros((tinggi, lebar, 3), dtype=np.uint8)
    np.random.seed(42)
    colors = [np.random.randint(60, 255, 3).tolist() for _ in range(jumlah_objek + 2)]
    for i in range(2, jumlah_objek + 1):
        marker_visual[markers == i] = colors[i]

    # Tahap 7: Algoritma Watershed
    markers_ws = markers.copy()
    markers_ws = cv2.watershed(img_bgr.copy(), markers_ws)

    # Batas Watershed (-1) diberi warna merah menyala pada citra asli
    hasil_overlay = img_rgb.copy()
    hasil_overlay[markers_ws == -1] = [255, 0, 0]

    # Tahap 8: Visualisasi Label Wilayah Berwarna
    hasil_label = np.zeros((tinggi, lebar, 3), dtype=np.uint8)
    for i in np.unique(markers_ws):
        if i > 1:
            hasil_label[markers_ws == i] = colors[i % len(colors)]

    return {
        "img_rgb": img_rgb,
        "gray": gray,
        "blur": blur,
        "thresh": thresh,
        "opening": opening,
        "dist_transform": dist_norm,
        "sure_bg": sure_bg,
        "sure_fg": sure_fg,
        "unknown": unknown,
        "jumlah_objek": max(0, jumlah_objek - 1),
        "marker_visual": marker_visual,
        "hasil_overlay": hasil_overlay,
        "hasil_label": hasil_label,
        "markers_raw": markers_ws
    }


# ==============================================================================
# 6. GRAPH CUT SEGMENTATION (GRABCUT - 8 TAHAPAN)
# ==============================================================================

def proses_grabcut_segmentation(
    img_bgr: np.ndarray,
    rect: Optional[Tuple[int, int, int, int]] = None,
    margin_ratio: float = 0.05,
    iterasi: int = 5
) -> Dict[str, Any]:
    """
    Eksekusi 8 Tahapan Teori GrabCut:
    Tahap 1: Membaca Citra & Preprocessing Gaussian Blur
    Tahap 2: Menentukan Bounding Box Rectangle
    Tahap 3: Inisialisasi Mask Foreground & Background
    Tahap 4: Pemodelan Warna GMM (Gaussian Mixture Model)
    Tahap 5: Menghitung Probabilitas Piksel (4 State Classification)
    Tahap 6 & 7: Optimasi Graph Cut & Iterasi Konvergensi
    Tahap 8: Hasil Segmentasi, Alpha Transparent, & Boundary Contours
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tinggi, lebar = img_bgr.shape[:2]

    # Tahap 1: Preprocessing
    img_blur = cv2.GaussianBlur(img_bgr, (3, 3), 0)

    # Tahap 2: Bounding Box
    if rect is None:
        margin_x = int(lebar * margin_ratio)
        margin_y = int(tinggi * margin_ratio)
        x = max(1, margin_x)
        y = max(1, margin_y)
        w = max(1, lebar - (2 * margin_x))
        h = max(1, tinggi - (2 * margin_y))
        rect = (x, y, w, h)
    else:
        x, y, w, h = rect

    img_bbox = img_rgb.copy()
    cv2.rectangle(img_bbox, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Tahap 3: Inisialisasi Mask
    mask = np.zeros((tinggi, lebar), dtype=np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Inisialisasi GC_INIT_WITH_RECT (1 iterasi awal)
    cv2.grabCut(img_blur, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
    mask_awal = mask.copy()

    # Tahap 4: Pemodelan GMM Visual
    gmm_visual = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # Tahap 5: Probabilitas Piksel (4 Tingkat Abu-abu)
    probabilitas_visual = np.zeros((tinggi, lebar), dtype=np.uint8)
    probabilitas_visual[mask == cv2.GC_BGD] = 0
    probabilitas_visual[mask == cv2.GC_PR_BGD] = 85
    probabilitas_visual[mask == cv2.GC_PR_FGD] = 170
    probabilitas_visual[mask == cv2.GC_FGD] = 255

    # Tahap 6 & 7: Optimasi Graph Cut & Iterasi
    hasil_iterasi_masks = [mask.copy()]
    for _ in range(max(0, iterasi - 1)):
        cv2.grabCut(img_blur, mask, None, bgdModel, fgdModel, 1, cv2.GC_EVAL)
        hasil_iterasi_masks.append(mask.copy())

    # Mask Final Biner
    mask_final = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # Tahap 8: Citra Segmentasi & Transparansi
    hasil_segmentasi_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_final)
    hasil_segmentasi_rgb = cv2.cvtColor(hasil_segmentasi_bgr, cv2.COLOR_BGR2RGB)

    # Citra Transparan RGBA
    img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
    img_rgba[:, :, 3] = mask_final

    # Garis Kontur Batas Objek
    hasil_batas = img_rgb.copy()
    contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(hasil_batas, contours, -1, (255, 0, 0), 2)

    return {
        "img_rgb": img_rgb,
        "rect": rect,
        "img_bbox": img_bbox,
        "mask_awal": mask_awal,
        "gmm_visual": gmm_visual,
        "probabilitas_visual": probabilitas_visual,
        "mask_final": mask_final,
        "hasil_segmentasi_rgb": hasil_segmentasi_rgb,
        "img_rgba": img_rgba,
        "hasil_batas": hasil_batas,
        "iterations_done": len(hasil_iterasi_masks)
    }
