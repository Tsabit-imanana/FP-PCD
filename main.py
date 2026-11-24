"""
Main CLI Script - PCD Final Project (Pemrosesan Citra Digital)
Kelompok 9 - UPN "Veteran" Jawa Timur

Menjalankan 6 metode segmentasi citra:
1. Otsu Thresholding & Background Removal
2. Edge-Based Segmentation (Canny Edge - 5 Tahapan Teori)
3. Region-Based Segmentation (Region Growing - 5 Tahapan Teori)
4. Clustering-Based Segmentation (K-Means - 7 Tahapan Teori)
5. Watershed Segmentation (8 Tahapan Teori)
6. Graph Cut Segmentation (GrabCut - 8 Tahapan Teori)
"""

import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# Import modul segmentasi
from pcd_segmentation import (
    proses_otsu_thresholding,
    proses_canny_edge,
    proses_region_growing,
    proses_kmeans_clustering,
    proses_watershed_segmentation,
    proses_grabcut_segmentation
)

FORMAT_GAMBAR = (
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp", ".jfif"
)


def temukan_gambar(folder: str) -> List[str]:
    """Mencari seluruh file citra yang valid dalam direktori."""
    daftar = []
    if not os.path.exists(folder):
        return daftar
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(FORMAT_GAMBAR):
                daftar.append(os.path.join(root, file))
    return sorted(daftar)


# ==============================================================================
# PIPELINE EKSEKUSI SETIAP METODE
# ==============================================================================

def jalankan_otsu(daftar_gambar: List[str], folder_output: str = "hasil_otsu"):
    """Menjalankan Otsu Thresholding & Background Removal."""
    os.makedirs(folder_output, exist_ok=True)
    print(f"\n[1/6] Menjalankan Otsu Thresholding & Background Removal ({len(daftar_gambar)} gambar)...")

    for i, path_img in enumerate(daftar_gambar, 1):
        nama = os.path.splitext(os.path.basename(path_img))[0]
        img_bgr = cv2.imread(path_img)
        if img_bgr is None:
            print(f"  Gagal membaca {path_img}, dilewati!")
            continue

        res = proses_otsu_thresholding(img_bgr)

        # Simpan file keluaran
        cv2.imwrite(os.path.join(folder_output, f"{nama}_threshold.png"), res["raw_thresh"])
        cv2.imwrite(os.path.join(folder_output, f"{nama}_mask.png"), res["mask_final"])
        cv2.imwrite(os.path.join(folder_output, f"{nama}_transparent.png"), cv2.cvtColor(res["transparent_rgba"], cv2.COLOR_RGBA2BGRA))

        # Visualisasi Matplotlib
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        axes[0].imshow(res["img_rgb"]); axes[0].set_title("Citra Asli")
        axes[1].imshow(res["gray"], cmap="gray"); axes[1].set_title("Grayscale")
        axes[2].imshow(res["raw_thresh"], cmap="gray"); axes[2].set_title(f"Otsu (T={res['threshold_value']:.1f})")
        axes[3].imshow(res["mask_final"], cmap="gray"); axes[3].set_title("Mask Optimasi")
        axes[4].imshow(res["transparent_rgba"]); axes[4].set_title("Background Dihapus")
        for ax in axes: ax.axis("off")
        plt.suptitle(f"Otsu Thresholding - {nama}", fontsize=13)
        plt.tight_layout()
        fig.savefig(os.path.join(folder_output, f"{nama}_plot.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {nama} berhasil diproses. Threshold: {res['threshold_value']:.1f}")


def jalankan_canny(daftar_gambar: List[str], folder_output: str = "hasil_edge_5_tahap"):
    """Menjalankan Canny Edge Detection (5 Tahapan Teori)."""
    os.makedirs(folder_output, exist_ok=True)
    print(f"\n[2/6] Menjalankan Edge-Based Canny 5 Tahapan ({len(daftar_gambar)} gambar)...")

    for i, path_img in enumerate(daftar_gambar, 1):
        nama = os.path.splitext(os.path.basename(path_img))[0]
        img_bgr = cv2.imread(path_img)
        if img_bgr is None: continue

        res = proses_canny_edge(img_bgr)

        cv2.imwrite(os.path.join(folder_output, f"{nama}_canny_edge.png"), res["edges"])
        cv2.imwrite(os.path.join(folder_output, f"{nama}_grad_mag.png"), res["gradient_magnitude"])

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(res["img_rgb"]); axes[0].set_title("Citra Asli")
        axes[1].imshow(res["gray"], cmap="gray"); axes[1].set_title("1. Grayscale")
        axes[2].imshow(res["blur"], cmap="gray"); axes[2].set_title("2. Gaussian Blur")
        axes[3].imshow(res["gradient_magnitude"], cmap="gray"); axes[3].set_title("3. Deteksi Gradien")
        axes[4].imshow(res["edges"], cmap="gray"); axes[4].set_title(f"4 & 5. NMS & Hysteresis\n({res['threshold_bawah']}, {res['threshold_atas']})")
        for ax in axes: ax.axis("off")
        plt.suptitle(f"Edge-Based Segmentation - {nama}", fontsize=13)
        plt.tight_layout()
        fig.savefig(os.path.join(folder_output, f"{nama}_plot.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {nama} berhasil diproses. Threshold: [{res['threshold_bawah']}, {res['threshold_atas']}]")


def jalankan_region_growing(daftar_gambar: List[str], folder_output: str = "hasil_region_growing"):
    """Menjalankan Region Growing (5 Tahapan Teori)."""
    os.makedirs(folder_output, exist_ok=True)
    print(f"\n[3/6] Menjalankan Region-Based Region Growing ({len(daftar_gambar)} gambar)...")

    for i, path_img in enumerate(daftar_gambar, 1):
        nama = os.path.splitext(os.path.basename(path_img))[0]
        img_bgr = cv2.imread(path_img)
        if img_bgr is None: continue

        res = proses_region_growing(img_bgr, threshold=30)

        cv2.imwrite(os.path.join(folder_output, f"{nama}_region_growing.png"), res["mask_region"])

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(res["img_rgb"]); axes[0].set_title("Citra Asli")
        axes[1].imshow(res["gray"], cmap="gray"); axes[1].set_title("1. Karakteristik (Grayscale)")
        axes[2].imshow(res["img_seed_viz"]); axes[2].set_title(f"2. Titik Awal Seed\n{res['seed_point']}")
        axes[3].imshow(res["blur"], cmap="gray"); axes[3].set_title("3 & 4. Cek Tetangga & Gabung")
        axes[4].imshow(res["mask_region"], cmap="gray"); axes[4].set_title(f"5. Wilayah ({res['area_percentage']}%)")
        for ax in axes: ax.axis("off")
        plt.suptitle(f"Region Growing - {nama}", fontsize=13)
        plt.tight_layout()
        fig.savefig(os.path.join(folder_output, f"{nama}_plot.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {nama} berhasil diproses. Piksel tumbuh: {res['pixel_count']} ({res['area_percentage']}%)")


def jalankan_kmeans(daftar_gambar: List[str], folder_output: str = "hasil_kmeans", K: int = 3):
    """Menjalankan K-Means Clustering (7 Tahapan Teori)."""
    os.makedirs(folder_output, exist_ok=True)
    print(f"\n[4/6] Menjalankan Clustering K-Means (K={K}) ({len(daftar_gambar)} gambar)...")

    for i, path_img in enumerate(daftar_gambar, 1):
        nama = os.path.splitext(os.path.basename(path_img))[0]
        img_bgr = cv2.imread(path_img)
        if img_bgr is None: continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = proses_kmeans_clustering(img_rgb, K=K)

        cv2.imwrite(os.path.join(folder_output, f"{nama}_kmeans.png"), cv2.cvtColor(res["segmented"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(folder_output, f"{nama}_label.png"), res["label_visual"])

        fig, axes = plt.subplots(2, 4, figsize=(20, 9))
        axes[0, 0].imshow(img_rgb); axes[0, 0].set_title("Citra Asli")
        axes[0, 1].imshow(img_rgb); axes[0, 1].set_title(f"1. Tentukan K={K}")
        axes[0, 2].imshow(res["palette_awal"]); axes[0, 2].set_title("2. Centroid Awal (K-Means++)")
        axes[0, 3].imshow(res["label_pertama_visual"], cmap="nipy_spectral"); axes[0, 3].set_title("3 & 4. Jarak & Partisi Awal")
        axes[1, 0].imshow(res["palette_akhir"]); axes[1, 0].set_title("5. Centroid Akhir")
        axes[1, 1].imshow(res["label_visual"], cmap="nipy_spectral"); axes[1, 1].set_title("6. Label Konvergen")
        axes[1, 2].imshow(res["segmented"]); axes[1, 2].set_title("7. Citra Rekonstruksi")

        # Bar chart distribusi piksel
        bar_colors = [tuple(c / 255.0) for c in res["centers_akhir"]]
        axes[1, 3].bar(range(1, K + 1), res["persentase_piksel"], color=bar_colors, edgecolor="black")
        axes[1, 3].set_title("Distribusi Piksel (%)")
        axes[1, 3].set_xlabel("Cluster")
        axes[1, 3].set_ylabel("% Area")

        for ax in axes.flatten()[:7]: ax.axis("off")
        plt.suptitle(f"K-Means Clustering - {nama} (Compactness: {res['compactness']:.1f})", fontsize=13)
        plt.tight_layout()
        fig.savefig(os.path.join(folder_output, f"{nama}_plot.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {nama} berhasil diproses. Compactness: {res['compactness']:.1f}")


def jalankan_watershed(daftar_gambar: List[str], folder_output: str = "hasil_watershed"):
    """Menjalankan Watershed Segmentation (8 Tahapan Teori)."""
    os.makedirs(folder_output, exist_ok=True)
    print(f"\n[5/6] Menjalankan Watershed Segmentation ({len(daftar_gambar)} gambar)...")

    for i, path_img in enumerate(daftar_gambar, 1):
        nama = os.path.splitext(os.path.basename(path_img))[0]
        img_bgr = cv2.imread(path_img)
        if img_bgr is None: continue

        res = proses_watershed_segmentation(img_bgr)

        cv2.imwrite(os.path.join(folder_output, f"{nama}_watershed.png"), cv2.cvtColor(res["hasil_overlay"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(folder_output, f"{nama}_label.png"), cv2.cvtColor(res["hasil_label"], cv2.COLOR_RGB2BGR))

        fig, axes = plt.subplots(2, 4, figsize=(20, 9))
        axes[0, 0].imshow(res["img_rgb"]); axes[0, 0].set_title("Citra Asli")
        axes[0, 1].imshow(res["blur"], cmap="gray"); axes[0, 1].set_title("1. Grayscale & Blur")
        axes[0, 2].imshow(res["thresh"], cmap="gray"); axes[0, 2].set_title("2. Threshold Inverted")
        axes[0, 3].imshow(res["opening"], cmap="gray"); axes[0, 3].set_title("3. Morfologi Opening")
        axes[1, 0].imshow(res["sure_fg"], cmap="gray"); axes[1, 0].set_title("4. Sure Foreground")
        axes[1, 1].imshow(res["unknown"], cmap="gray"); axes[1, 1].set_title("5. Unknown Region")
        axes[1, 2].imshow(res["marker_visual"]); axes[1, 2].set_title(f"6 & 7. Marker ({res['jumlah_objek']} Objek)")
        axes[1, 3].imshow(res["hasil_overlay"]); axes[1, 3].set_title("8. Hasil Batas Watershed")
        for ax in axes.flatten(): ax.axis("off")
        plt.suptitle(f"Watershed Segmentation - {nama}", fontsize=13)
        plt.tight_layout()
        fig.savefig(os.path.join(folder_output, f"{nama}_plot.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {nama} berhasil diproses. Objek terdeteksi: {res['jumlah_objek']}")


def jalankan_grabcut(daftar_gambar: List[str], folder_output: str = "hasil_grabcut", iterasi: int = 5):
    """Menjalankan GrabCut Segmentation (8 Tahapan Teori)."""
    os.makedirs(folder_output, exist_ok=True)
    print(f"\n[6/6] Menjalankan GrabCut Segmentation ({len(daftar_gambar)} gambar)...")

    for i, path_img in enumerate(daftar_gambar, 1):
        nama = os.path.splitext(os.path.basename(path_img))[0]
        img_bgr = cv2.imread(path_img)
        if img_bgr is None: continue

        res = proses_grabcut_segmentation(img_bgr, iterasi=iterasi)

        cv2.imwrite(os.path.join(folder_output, f"{nama}_grabcut.png"), cv2.cvtColor(res["hasil_segmentasi_rgb"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(folder_output, f"{nama}_mask.png"), res["mask_final"])
        cv2.imwrite(os.path.join(folder_output, f"{nama}_transparan.png"), res["img_rgba"])

        fig, axes = plt.subplots(2, 4, figsize=(20, 9))
        axes[0, 0].imshow(res["img_rgb"]); axes[0, 0].set_title("Citra Asli")
        axes[0, 1].imshow(res["img_bbox"]); axes[0, 1].set_title(f"1 & 2. Bounding Box {res['rect']}")
        axes[0, 2].imshow(res["mask_awal"], cmap="gray"); axes[0, 2].set_title("3. Inisialisasi Mask")
        axes[0, 3].imshow(res["gmm_visual"], cmap="gray"); axes[0, 3].set_title("4. Model GMM")
        axes[1, 0].imshow(res["probabilitas_visual"], cmap="gray"); axes[1, 0].set_title("5. Probabilitas Piksel")
        axes[1, 1].imshow(res["mask_final"], cmap="gray"); axes[1, 1].set_title(f"6 & 7. Graph Cut ({iterasi} Itr)")
        axes[1, 2].imshow(res["hasil_segmentasi_rgb"]); axes[1, 2].set_title("8. Hasil Segmentasi")
        axes[1, 3].imshow(res["hasil_batas"]); axes[1, 3].set_title("8. Kontur Batas Objek")
        for ax in axes.flatten(): ax.axis("off")
        plt.suptitle(f"GrabCut Segmentation - {nama}", fontsize=13)
        plt.tight_layout()
        fig.savefig(os.path.join(folder_output, f"{nama}_plot.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {nama} berhasil diproses ({iterasi} iterasi).")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Aplikasi CLI Segmentasi PCD - Kelompok 9 PCD UPN 'Veteran' Jawa Timur"
    )
    parser.add_argument(
        "--input", "-i",
        default="sample_images",
        help="Folder direktori citra masukan atau path satu file citra (default: sample_images)"
    )
    parser.add_argument(
        "--metode", "-m",
        default="all",
        choices=["all", "otsu", "canny", "region", "kmeans", "watershed", "grabcut"],
        help="Metode segmentasi yang dijalankan (default: all)"
    )
    parser.add_argument("--k", type=int, default=3, help="Jumlah cluster K untuk K-Means (default: 3)")
    parser.add_argument("--iterasi", type=int, default=5, help="Jumlah iterasi untuk GrabCut (default: 5)")
    args = parser.parse_args()

    # Periksa input gambar
    if os.path.isfile(args.input):
        daftar_gambar = [args.input]
    else:
        daftar_gambar = temukan_gambar(args.input)

    if not daftar_gambar:
        print(f"[ERROR] Tidak ada citra yang ditemukan di: '{args.input}'")
        print("Pastikan folder memiliki file berformat .jpg, .png, .jfif, .webp, dll.")
        sys.exit(1)

    print("=================================================================")
    print(" PENGOLAHAN CITRA DIGITAL (PCD) - IMPLEMENTASI METODE SEGMENTASI")
    print(" Kelompok 9 - UPN 'Veteran' Jawa Timur")
    print("=================================================================")
    print(f"Direktori Citra Masukan : {args.input}")
    print(f"Jumlah Citra Ditemukan  : {len(daftar_gambar)}")
    print(f"Metode Yang Dipilih     : {args.metode.upper()}")
    print("=================================================================")

    if args.metode in ["all", "otsu"]:
        jalankan_otsu(daftar_gambar)
    if args.metode in ["all", "canny"]:
        jalankan_canny(daftar_gambar)
    if args.metode in ["all", "region"]:
        jalankan_region_growing(daftar_gambar)
    if args.metode in ["all", "kmeans"]:
        jalankan_kmeans(daftar_gambar, K=args.k)
    if args.metode in ["all", "watershed"]:
        jalankan_watershed(daftar_gambar)
    if args.metode in ["all", "grabcut"]:
        jalankan_grabcut(daftar_gambar, iterasi=args.iterasi)

    print("\n=================================================================")
    print("✓ SELURUH PROSES SEGMENTASI SELESAI!")
    print("=================================================================")


if __name__ == "__main__":
    main()
