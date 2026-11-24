"""
PCD Final Project - Interactive Web Application
Studi Komparasi 6 Metode Segmentasi Citra Digital
Berdasarkan Analisis PCD_FINAL.ipynb - Kelompok 9 PCD UPN "Veteran" Jawa Timur

Fitur Aplikasi:
1. Eksplorasi Interaktif Setiap Tahapan Algoritma Segmentasi
2. Live Parameter Adjustment (Sliders & Checkboxes)
3. Komparasi 6 Metode Sekaligus (Side-by-Side Comparison)
4. Batch Processing Folder Citra
5. Dokumentasi Teori Lengkap & Analisis PCD_FINAL.ipynb
"""

import os
import io
import time
import cv2
import numpy as np
# pyrefly: ignore[missing-import]
import streamlit as st
from PIL import Image

# Import library segmentasi yang telah dibuat
from pcd_segmentation import (
    proses_otsu_thresholding,
    proses_canny_edge,
    proses_region_growing,
    proses_kmeans_clustering,
    proses_watershed_segmentation,
    proses_grabcut_segmentation
)

# ==============================================================================
# KONFIGURASI HALAMAN & TEMA STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="PCD Segmentation Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        margin-bottom: 8px;
        font-size: 28px;
        font-weight: 700;
    }
    .main-header p {
        color: #e0f2fe;
        font-size: 15px;
        margin: 0;
    }
    .method-badge {
        display: inline-block;
        padding: 4px 12px;
        background-color: #2563eb;
        color: white;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    .metric-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .step-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px;
        background: #ffffff;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
@st.cache_data
def get_sample_images():
    """Mengambil daftar gambar contoh yang tersedia di folder sample_images."""
    sample_dir = "sample_images"
    if not os.path.exists(sample_dir):
        return {}
    files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".png", ".jfif", ".webp", ".jpeg"))]
    return {f.split(".")[0].capitalize(): os.path.join(sample_dir, f) for f in files}


def convert_image_to_bytes(img: np.ndarray, format: str = "PNG") -> bytes:
    """Mengonversi numpy array image ke byte buffer untuk tombol download."""
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return buf.getvalue()


# ==============================================================================
# HEADER UTAMA
# ==============================================================================
st.markdown("""
<div class="main-header">
    <h1>🔬 Studio Segmentasi Citra Digital (PCD)</h1>
    <p>Implementasi & Komparasi 6 Algoritma Segmentasi Citra | Diadaptasi dari <b>PCD_FINAL.ipynb</b> (Kelompok 9 - PCD UPN "Veteran" Jatim)</p>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR: PENGATURAN CITRA MASUKAN
# ==============================================================================
st.sidebar.header("⚙️ Pengaturan Citra Masukan")

sample_dict = get_sample_images()
pilihan_sumber = st.sidebar.radio(
    "Sumber Citra:",
    ["🖼️ Gambar Sampel Dataset", "📤 Unggah Gambar Sendiri"]
)

img_bgr = None
nama_citra = "citra_pcd"

if pilihan_sumber == "🖼️ Gambar Sampel Dataset":
    if sample_dict:
        sample_name = st.sidebar.selectbox("Pilih Sampel:", list(sample_dict.keys()))
        sample_path = sample_dict[sample_name]
        nama_citra = sample_name.lower()
        img_bgr = cv2.imread(sample_path)
    else:
        st.sidebar.warning("Folder 'sample_images' belum memiliki gambar.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Unggah Citra (JPG, PNG, JPEG, WEBP):",
        type=["jpg", "jpeg", "png", "webp", "bmp"]
    )
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        nama_citra = os.path.splitext(uploaded_file.name)[0]

# Fallback generator jika belum ada gambar yang dimuat
if img_bgr is None:
    # Buat gambar sintetis sederhana untuk preview
    h, w = 300, 300
    synthetic = np.zeros((h, w, 3), dtype=np.uint8) + 240
    cv2.circle(synthetic, (150, 150), 80, (50, 120, 220), -1)
    cv2.rectangle(synthetic, (50, 50), (100, 100), (40, 180, 70), -1)
    img_bgr = synthetic
    nama_citra = "sintetis"
    st.info("💡 Menampilkan citra sintetis sebagai demo awal. Silakan pilih gambar sampel atau unggah citra Anda di sidebar.")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_bgr.shape[:2]

with st.sidebar.expander("ℹ️ Info Citra Aktif", expanded=True):
    st.write(f"**Nama:** `{nama_citra}`")
    st.write(f"**Resolusi:** `{w} x {h}` piksel")
    st.write(f"**Total Piksel:** `{w * h:,}`")
    st.image(img_rgb, caption="Citra Aktif", use_container_width=True)


# ==============================================================================
# TAB NAVIGASI UTAMA
# ==============================================================================
tab_eksplorasi, tab_komparasi, tab_analisis, tab_batch = st.tabs([
    "🧪 Eksplorasi Tahapan Metode",
    "⚖️ Komparasi 6 Metode Sekaligus",
    "📖 Analisis Teori & Notebook",
    "📁 Batch Processing"
])


# ==============================================================================
# TAB 1: EKSPLORASI TAHAPAN METODE (INTERAKTIF)
# ==============================================================================
with tab_eksplorasi:
    metode = st.selectbox(
        "Pilih Metode Segmentasi yang Ingin Dieksplorasi:",
        [
            "1. Otsu Thresholding & Background Removal (Cell 0)",
            "2. Edge-Based Segmentation - Canny 5 Tahap (Cell 1)",
            "3. Region-Based Segmentation - Region Growing 5 Tahap (Cell 2)",
            "4. Clustering-Based Segmentation - K-Means 7 Tahap (Cell 3)",
            "5. Watershed Segmentation - 8 Tahap (Cell 4)",
            "6. Graph Cut Segmentation - GrabCut 8 Tahap (Cell 5)"
        ]
    )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 1. OTSU THRESHOLDING
    # --------------------------------------------------------------------------
    if "Otsu" in metode:
        st.subheader("1. Otsu Thresholding & Background Removal Pipeline")
        st.markdown(r"""
        *Metode thresholding adaptif global berdasarkan pemaksimalan varians antar-kelas ($\sigma_B^2$), 
        dilengkapi optimasi morfologi opening/closing, seleksi komponen terhubung terbesar, flood fill hole filling, dan penghalusan tepi alpha.*
        """)

        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            blur_k = st.slider("Gaussian Blur Kernel:", 1, 15, 5, step=2)
        with col_p2:
            morph_k = st.slider("Morfologi Kernel:", 1, 15, 5, step=2)
        with col_p3:
            morph_iter = st.slider("Iterasi Morfologi:", 1, 5, 2)
        with col_p4:
            auto_inv = st.checkbox("Auto Inversi (Deteksi Tepi)", value=True)
            keep_largest = st.checkbox("Simpan Objek Terbesar", value=True)
            fill_holes = st.checkbox("Hole Filling (Flood Fill)", value=True)
            smooth = st.checkbox("Haluskan Tepi Mask", value=True)

        t_start = time.time()
        res_otsu = proses_otsu_thresholding(
            img_bgr,
            blur_ksize=blur_k,
            morph_ksize=morph_k,
            morph_iters=morph_iter,
            auto_invert=auto_inv,
            keep_largest_component=keep_largest,
            fill_holes=fill_holes,
            smooth_mask=smooth
        )
        elapsed = time.time() - t_start

        # Metrik
        fg_pixels = np.count_nonzero(res_otsu["mask_final"] > 127)
        fg_percent = round((fg_pixels / (w * h)) * 100, 2)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Threshold Nilai (T)", f"{res_otsu['threshold_value']:.1f}")
        m2.metric("Inversi Dilakukan?", "Ya" if res_otsu['inverted'] else "Tidak")
        m3.metric("Area Objek (%)", f"{fg_percent}%")
        m4.metric("Waktu Eksekusi", f"{elapsed*1000:.1f} ms")

        # Visualisasi 5 Tahap
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.image(res_otsu["img_rgb"], caption="1. Citra Asli", use_container_width=True)
        with c2:
            st.image(res_otsu["gray"], caption="2. Grayscale", use_container_width=True)
        with c3:
            st.image(res_otsu["raw_thresh"], caption=f"3. Otsu (T={res_otsu['threshold_value']:.0f})", use_container_width=True)
        with c4:
            st.image(res_otsu["mask_final"], caption="4. Mask Optimasi", use_container_width=True)
        with c5:
            st.image(res_otsu["transparent_rgba"], caption="5. Background Dihapus", use_container_width=True)

        st.markdown("#### 📥 Unduh Hasil")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button(
                "⬇️ Unduh Mask Biner (PNG)",
                data=convert_image_to_bytes(res_otsu["mask_final"]),
                file_name=f"{nama_citra}_otsu_mask.png",
                mime="image/png"
            )
        with d_col2:
            st.download_button(
                "⬇️ Unduh Citra Transparan (RGBA PNG)",
                data=convert_image_to_bytes(res_otsu["transparent_rgba"]),
                file_name=f"{nama_citra}_otsu_transparent.png",
                mime="image/png"
            )

    # --------------------------------------------------------------------------
    # 2. CANNY EDGE DETECTION
    # --------------------------------------------------------------------------
    elif "Canny" in metode:
        st.subheader("2. Edge-Based Segmentation (Canny Edge - 5 Tahapan Teori)")
        st.markdown("""
        *Implementasi 5 tahapan teori Canny Edge Detection: 
        1. Grayscale → 2. Gaussian Blur Noise Reduction → 3. Deteksi Gradien (Sobel X/Y & Magnitude) → 
        4. Non-Maximum Suppression (NMS) → 5. Hysteresis Thresholding.*
        """)

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            blur_k = st.slider("Gaussian Blur Kernel:", 1, 15, 5, step=2)
            blur_sigma = st.slider("Gaussian Sigma (σ):", 0.5, 3.0, 1.4, step=0.1)
        with col_p2:
            auto_thresh = st.checkbox("Gunakan Adaptive Threshold Otomatis (Median Base)", value=True)
            sigma_ratio = st.slider("Sigma Ratio:", 0.1, 0.6, 0.33, step=0.01, disabled=not auto_thresh)
        with col_p3:
            low_t = st.slider("Manual Threshold Bawah:", 0, 255, 50, disabled=auto_thresh)
            high_t = st.slider("Manual Threshold Atas:", 0, 255, 150, disabled=auto_thresh)

        t_start = time.time()
        res_canny = proses_canny_edge(
            img_bgr,
            blur_ksize=blur_k,
            blur_sigma=blur_sigma,
            auto_threshold=auto_thresh,
            low_thresh=low_t,
            high_thresh=high_t,
            sigma_ratio=sigma_ratio
        )
        elapsed = time.time() - t_start

        edge_pixels = np.count_nonzero(res_canny["edges"] > 0)
        edge_percent = round((edge_pixels / (w * h)) * 100, 2)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Threshold Bawah", f"{res_canny['threshold_bawah']}")
        m2.metric("Threshold Atas", f"{res_canny['threshold_atas']}")
        m3.metric("Kepadatan Tepi", f"{edge_percent}% ({edge_pixels:,} px)")
        m4.metric("Waktu Eksekusi", f"{elapsed*1000:.1f} ms")

        # 5 Tahapan Teori Notebook
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.image(res_canny["img_rgb"], caption="Citra Asli", use_container_width=True)
        with c2:
            st.image(res_canny["gray"], caption="1. Grayscale", use_container_width=True)
        with c3:
            st.image(res_canny["blur"], caption=f"2. Blur (σ={blur_sigma})", use_container_width=True)
        with c4:
            st.image(res_canny["gradient_magnitude"], caption="3. Deteksi Gradien (Sobel)", use_container_width=True)
        with c5:
            st.image(res_canny["edges"], caption=f"4 & 5. NMS & Hysteresis", use_container_width=True)

        st.markdown("#### 🔍 Visualisasi Overlay Tepi pada Citra Asli")
        col_ov1, col_ov2 = st.columns(2)
        with col_ov1:
            st.image(res_canny["overlay"], caption="Overlay Tepi Canny (Garis Merah)", use_container_width=True)
        with col_ov2:
            st.markdown(r"""
            **Analisis Tahapan Teori Canny:**
            - **Sobel Magnitude:** Menangkap perubahan intensitas piksel pertama ($\sqrt{G_x^2 + G_y^2}$).
            - **Non-Maximum Suppression (NMS):** Menipiskan tepi dengan mempertahankan piksel bernilai maksimum lokal sepanjang arah gradien.
            - **Hysteresis Thresholding:** Menghubungkan tepi kuat (di atas threshold atas) dan tepi lemah (antara threshold bawah dan atas) jika terhubung secara spasial.
            """)
            st.download_button(
                "⬇️ Unduh Citra Tepi Canny (PNG)",
                data=convert_image_to_bytes(res_canny["edges"]),
                file_name=f"{nama_citra}_canny_edge.png",
                mime="image/png"
            )

    # --------------------------------------------------------------------------
    # 3. REGION GROWING
    # --------------------------------------------------------------------------
    elif "Region" in metode:
        st.subheader("3. Region-Based Segmentation (Region Growing - 5 Tahapan Teori)")
        st.markdown(r"""
        *Pendekatan berbasis wilayah yang menumbuhkan region dari **Seed Point (titik awal)** 
        menggunakan antrean Breadth-First Search (BFS) berdasarkan kriteria kemiripan intensitas piksel tetangga ($|I(x,y) - I(seed)| \le \Delta T$).*
        """)

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            seed_mode = st.radio("Posisi Titik Awal (Seed Point):", ["Tengah Citra (Default Notebook)", "Kustom Manual (X, Y)"])
            if seed_mode == "Tengah Citra (Default Notebook)":
                seed_x, seed_y = w // 2, h // 2
            else:
                seed_x = st.slider("Koordinat X:", 0, w - 1, w // 2)
                seed_y = st.slider("Koordinat Y:", 0, h - 1, h // 2)
        with col_p2:
            thresh_grow = st.slider("Threshold Kesamaan Intensitas (ΔT):", 5, 80, 30)
            blur_k = st.slider("Kernel Penghalusan:", 1, 15, 5, step=2)
        with col_p3:
            konektivitas = st.radio("Konektivitas Tetangga:", [8, 4], index=0)

        t_start = time.time()
        res_region = proses_region_growing(
            img_bgr,
            seed_point=(seed_x, seed_y),
            threshold=thresh_grow,
            connectivity=konektivitas,
            blur_ksize=blur_k
        )
        elapsed = time.time() - t_start

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Titik Awal (Seed)", f"({seed_x}, {seed_y})")
        m2.metric("Intensitas Seed", f"{res_region['seed_value']}")
        m3.metric("Piksel Tumbuh", f"{res_region['pixel_count']:,} ({res_region['area_percentage']}%)")
        m4.metric("Waktu Eksekusi", f"{elapsed*1000:.1f} ms")

        # 5 Tahapan Teori
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.image(res_region["img_rgb"], caption="Citra Asli", use_container_width=True)
        with c2:
            st.image(res_region["gray"], caption="1. Karakteristik (Grayscale)", use_container_width=True)
        with c3:
            st.image(res_region["img_seed_viz"], caption=f"2. Titik Seed ({seed_x},{seed_y})", use_container_width=True)
        with c4:
            st.image(res_region["blur"], caption="3 & 4. Cek Tetangga (BFS)", use_container_width=True)
        with c5:
            st.image(res_region["mask_region"], caption=f"5. Wilayah Tersegmentasi", use_container_width=True)

        st.markdown("#### 🔍 Hasil Segmentasi & Blended Overlay")
        col_rg1, col_rg2 = st.columns(2)
        with col_rg1:
            st.image(res_region["hasil_segmented"], caption="Foreground Wilayah Tersegmentasi", use_container_width=True)
        with col_rg2:
            st.image(res_region["blended"], caption="Highlight Overlay (Kuning)", use_container_width=True)
            st.download_button(
                "⬇️ Unduh Mask Region (PNG)",
                data=convert_image_to_bytes(res_region["mask_region"]),
                file_name=f"{nama_citra}_region_growing.png",
                mime="image/png"
            )

    # --------------------------------------------------------------------------
    # 4. K-MEANS CLUSTERING
    # --------------------------------------------------------------------------
    elif "K-Means" in metode:
        st.subheader("4. Clustering-Based Segmentation (K-Means - 7 Tahapan Teori)")
        st.markdown("""
        *Pengelompokan piksel citra pada ruang warna RGB ke dalam **K cluster** menggunakan algoritma K-Means++.
        Menampilkan inisialisasi centroid awal, perhitungan jarak Euclidean, pembaruan centroid, dan konvergensi.*
        """)

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            k_val = st.slider("Jumlah Cluster (K):", 2, 8, 3)
        with col_p2:
            max_itr = st.slider("Maksimal Iterasi:", 10, 200, 100, step=10)
        with col_p3:
            eps = st.number_input("Kriteria Epsilon:", value=1.0, min_value=0.1, max_value=5.0, step=0.1)

        t_start = time.time()
        res_km = proses_kmeans_clustering(img_rgb, K=k_val, max_iter=max_itr, epsilon=eps)
        elapsed = time.time() - t_start

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Jumlah Cluster (K)", f"{k_val}")
        m2.metric("Nilai Compactness", f"{res_km['compactness']:,.0f}")
        m3.metric("Iterasi / Epsilon", f"{max_itr} / {eps}")
        m4.metric("Waktu Komputasi", f"{elapsed*1000:.1f} ms")

        # 7 Tahapan Teori Notebook (2 Baris)
        st.markdown("#### 📊 Visualisasi 7 Tahapan K-Means")
        row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)
        with row1_c1:
            st.image(img_rgb, caption="Citra Asli", use_container_width=True)
        with row1_c2:
            st.image(img_rgb, caption=f"1. Tentukan Cluster K={k_val}", use_container_width=True)
        with row1_c3:
            st.image(res_km["palette_awal"], caption="2. Centroid Awal (K-Means++)", use_container_width=True)
        with row1_c4:
            st.image(res_km["label_pertama_visual"], caption="3 & 4. Jarak & Partisi Awal", use_container_width=True)

        row2_c1, row2_c2, row2_c3, row2_c4 = st.columns(4)
        with row2_c1:
            st.image(res_km["palette_akhir"], caption="5. Centroid Akhir", use_container_width=True)
        with row2_c2:
            st.image(res_km["label_visual"], caption="6. Label Konvergen", use_container_width=True)
        with row2_c3:
            st.image(res_km["segmented"], caption="7. Citra Rekonstruksi", use_container_width=True)
        with row2_c4:
            # Tampilkan palet cluster warna dan persentasenya
            st.markdown("**Distribusi Cluster Warna:**")
            for idx, (cnt, pct, center) in enumerate(zip(res_km["jumlah_piksel"], res_km["persentase_piksel"], res_km["centers_akhir"]), 1):
                rgb_code = tuple([int(c) for c in center])
                hex_code = f"#{rgb_code[0]:02x}{rgb_code[1]:02x}{rgb_code[2]:02x}"
                st.markdown(
                    f"<div style='display:flex; align-items:center; margin-bottom:4px;'>"
                    f"<div style='width:20px; height:20px; background-color:{hex_code}; border-radius:4px; margin-right:8px; border:1px solid #ccc;'></div>"
                    f"<span>Cluster {idx}: <b>{pct}%</b> ({cnt:,} px)</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.download_button(
            "⬇️ Unduh Citra Rekonstruksi K-Means (PNG)",
            data=convert_image_to_bytes(res_km["segmented"]),
            file_name=f"{nama_citra}_kmeans_k{k_val}.png",
            mime="image/png"
        )

    # --------------------------------------------------------------------------
    # 5. WATERSHED SEGMENTATION
    # --------------------------------------------------------------------------
    elif "Watershed" in metode:
        st.subheader("5. Watershed Segmentation (8 Tahapan Teori)")
        st.markdown("""
        *Algoritma morfologi topografi yang menganggap intensitas piksel sebagai ketinggian medan (relief).
        Menggunakan kombinasi Otsu Inverted, Morfologi Opening, Distance Transform untuk Sure Foreground & Sure Background, 
        serta Connected Component Marker.*
        """)

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            fg_r = st.slider("Threshold Sure Foreground Ratio:", 0.1, 0.8, 0.4, step=0.05)
        with col_p2:
            dil_itr = st.slider("Iterasi Dilasi Sure Background:", 1, 6, 3)
        with col_p3:
            open_itr = st.slider("Iterasi Opening Morfologi:", 1, 4, 2)

        t_start = time.time()
        res_ws = proses_watershed_segmentation(
            img_bgr,
            fg_ratio=fg_r,
            dilate_iters=dil_itr,
            open_iters=open_itr
        )
        elapsed = time.time() - t_start

        m1, m2, m3 = st.columns(3)
        m1.metric("Jumlah Objek Dideteksi", f"{res_ws['jumlah_objek']}")
        m2.metric("FG Threshold Ratio", f"{fg_r * 100:.0f}%")
        m3.metric("Waktu Eksekusi", f"{elapsed*1000:.1f} ms")

        # 8 Tahapan Teori Notebook (2 Baris x 4 Kolom)
        st.markdown("#### 🌊 Visualisasi 8 Tahapan Teori Watershed")
        r1_1, r1_2, r1_3, r1_4 = st.columns(4)
        with r1_1:
            st.image(res_ws["img_rgb"], caption="Citra Asli", use_container_width=True)
        with r1_2:
            st.image(res_ws["blur"], caption="1. Grayscale & Blur", use_container_width=True)
        with r1_3:
            st.image(res_ws["thresh"], caption="2. Otsu Inverted", use_container_width=True)
        with r1_4:
            st.image(res_ws["opening"], caption="3. Morfologi Opening", use_container_width=True)

        r2_1, r2_2, r2_3, r2_4 = st.columns(4)
        with r2_1:
            st.image(res_ws["sure_fg"], caption="4. Sure Foreground", use_container_width=True)
        with r2_2:
            st.image(res_ws["unknown"], caption="5. Unknown Region", use_container_width=True)
        with r2_3:
            st.image(res_ws["marker_visual"], caption=f"6 & 7. Marker ({res_ws['jumlah_objek']} Objek)", use_container_width=True)
        with r2_4:
            st.image(res_ws["hasil_overlay"], caption="8. Garis Batas Watershed", use_container_width=True)

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.image(res_ws["hasil_label"], caption="Label Wilayah Berwarna (Pseudo-color)", use_container_width=True)
        with col_w2:
            st.download_button(
                "⬇️ Unduh Batas Watershed (PNG)",
                data=convert_image_to_bytes(res_ws["hasil_overlay"]),
                file_name=f"{nama_citra}_watershed.png",
                mime="image/png"
            )

    # --------------------------------------------------------------------------
    # 6. GRABCUT SEGMENTATION
    # --------------------------------------------------------------------------
    elif "GrabCut" in metode:
        st.subheader("6. Graph Cut-Based Segmentation (GrabCut - 8 Tahapan Teori)")
        st.markdown("""
        *Algoritma interaktif berbasis optimasi Graph Cut dan Gaussian Mixture Model (GMM).
        Memisahkan objek utama dari background dengan batas bounding box persegi panjang.*
        """)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            margin_pct = st.slider("Margin Bounding Box (% dari tepi):", 1, 20, 5, step=1)
        with col_p2:
            iter_gc = st.slider("Jumlah Iterasi Graph Cut:", 1, 10, 5)

        t_start = time.time()
        res_gc = proses_grabcut_segmentation(
            img_bgr,
            margin_ratio=margin_pct / 100.0,
            iterasi=iter_gc
        )
        elapsed = time.time() - t_start

        fg_pixels = np.count_nonzero(res_gc["mask_final"] > 0)
        fg_pct = round((fg_pixels / (w * h)) * 100, 2)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bounding Box", f"{res_gc['rect']}")
        m2.metric("Iterasi Dijalankan", f"{iter_gc}")
        m3.metric("Area Foreground", f"{fg_pct}%")
        m4.metric("Waktu Eksekusi", f"{elapsed*1000:.1f} ms")

        # 8 Tahapan Teori Notebook
        st.markdown("#### ✂️ Visualisasi 8 Tahapan Teori GrabCut")
        r1_1, r1_2, r1_3, r1_4 = st.columns(4)
        with r1_1:
            st.image(res_gc["img_rgb"], caption="Citra Asli", use_container_width=True)
        with r1_2:
            st.image(res_gc["img_bbox"], caption=f"1 & 2. Bounding Box {res_gc['rect']}", use_container_width=True)
        with r1_3:
            st.image(res_gc["mask_awal"], caption="3. Inisialisasi Mask", use_container_width=True)
        with r1_4:
            st.image(res_gc["gmm_visual"], caption="4. Pemodelan GMM", use_container_width=True)

        r2_1, r2_2, r2_3, r2_4 = st.columns(4)
        with r2_1:
            st.image(res_gc["probabilitas_visual"], caption="5. Klasifikasi Probabilitas", use_container_width=True)
        with r2_2:
            st.image(res_gc["mask_final"], caption=f"6 & 7. Mask Akhir ({iter_gc} Itr)", use_container_width=True)
        with r2_3:
            st.image(res_gc["hasil_segmentasi_rgb"], caption="8. Citra Segmentasi", use_container_width=True)
        with r2_4:
            st.image(res_gc["hasil_batas"], caption="8. Garis Kontur Batas", use_container_width=True)

        st.markdown("#### 📥 Unduh Hasil GrabCut")
        gc_col1, gc_col2 = st.columns(2)
        with gc_col1:
            st.image(res_gc["img_rgba"], caption="Citra Transparan (Alpha Channel)", use_container_width=True)
        with gc_col2:
            st.download_button(
                "⬇️ Unduh Citra Transparan (RGBA PNG)",
                data=convert_image_to_bytes(res_gc["img_rgba"]),
                file_name=f"{nama_citra}_grabcut_transparan.png",
                mime="image/png"
            )


# ==============================================================================
# TAB 2: KOMPARASI 6 METODE SEKALIGUS (SIDE-BY-SIDE)
# ==============================================================================
with tab_komparasi:
    st.subheader("⚖️ Komparasi 6 Metode Segmentasi Citra Sekaligus")
    st.markdown("""
    Jalankan seluruh metode pada citra aktif dan bandingkan performa visual serta karakteristiknya secara langsung.
    """)

    if st.button("🚀 Jalankan Komparasi Lengkap (6 Metode)", type="primary"):
        with st.spinner("Sedang memproses seluruh metode segmentasi..."):
            # 1. Otsu
            t0 = time.time()
            r_otsu = proses_otsu_thresholding(img_bgr)
            t_otsu = (time.time() - t0) * 1000

            # 2. Canny
            t0 = time.time()
            r_canny = proses_canny_edge(img_bgr)
            t_canny = (time.time() - t0) * 1000

            # 3. Region Growing
            t0 = time.time()
            r_rg = proses_region_growing(img_bgr)
            t_rg = (time.time() - t0) * 1000

            # 4. K-Means
            t0 = time.time()
            r_km = proses_kmeans_clustering(img_rgb, K=3)
            t_km = (time.time() - t0) * 1000

            # 5. Watershed
            t0 = time.time()
            r_ws = proses_watershed_segmentation(img_bgr)
            t_ws = (time.time() - t0) * 1000

            # 6. GrabCut
            t0 = time.time()
            r_gc = proses_grabcut_segmentation(img_bgr, iterasi=5)
            t_gc = (time.time() - t0) * 1000

        st.success("✅ Seluruh 6 metode berhasil dieksekusi!")

        # Galeri Grid 6 Metode
        st.markdown("### 🖼️ Galeri Visual Hasil Akhir")
        g1, g2, g3 = st.columns(3)
        with g1:
            st.image(r_otsu["transparent_rgba"], caption=f"1. Otsu BG Removal ({t_otsu:.1f} ms)", use_container_width=True)
            st.image(r_km["segmented"], caption=f"4. K-Means Clustering K=3 ({t_km:.1f} ms)", use_container_width=True)
        with g2:
            st.image(r_canny["edges"], caption=f"2. Canny Edge Detection ({t_canny:.1f} ms)", use_container_width=True)
            st.image(r_ws["hasil_overlay"], caption=f"5. Watershed Overlay ({t_ws:.1f} ms)", use_container_width=True)
        with g3:
            st.image(r_rg["blended"], caption=f"3. Region Growing ({t_rg:.1f} ms)", use_container_width=True)
            st.image(r_gc["hasil_segmentasi_rgb"], caption=f"6. GrabCut Graph Cut ({t_gc:.1f} ms)", use_container_width=True)

        # Tabel Komparasi
        st.markdown("### 📋 Matriks Evaluasi & Karakteristik Metode")
        komparasi_data = {
            "Metode": [
                "Otsu Thresholding",
                "Canny Edge Detection",
                "Region Growing",
                "K-Means Clustering",
                "Watershed",
                "GrabCut"
            ],
            "Pendekatan Utama": [
                "Thresholding Global Intensitas",
                "Deteksi Tepi / Kontur Gradien",
                "Pertumbuhan Wilayah Spasial",
                "Clustering Ruang Warna (RGB)",
                "Morfologi Topografi Terarah",
                "Optimasi Energi Graph Cut"
            ],
            "Waktu Komputasi": [
                f"{t_otsu:.1f} ms",
                f"{t_canny:.1f} ms",
                f"{t_rg:.1f} ms",
                f"{t_km:.1f} ms",
                f"{t_ws:.1f} ms",
                f"{t_gc:.1f} ms"
            ],
            "Hasil Utama": [
                "Mask biner & citra transparan",
                "Peta garis tepi biner",
                "Mask wilayah kontinu",
                "Citra warna terkuantisasi",
                "Batas segmen batas objek",
                "Ekstraksi objek utama (RGBA)"
            ],
            "Skenario Terbaik": [
                "Objek kontras terhadap background homogen",
                "Deteksi batas bentuk geometri",
                "Objek terhubung dengan warna seragam",
                "Analisis distribusi warna multi-kelas",
                "Pemisahan objek bertumpuk / menempel",
                "Ekstraksi foreground alami kompleks"
            ]
        }
        st.table(komparasi_data)
    else:
        st.info("Klik tombol di atas untuk menjalankan komparasi pada citra aktif.")


# ==============================================================================
# TAB 3: ANALISIS TEORI & NOTEBOOK PCD_FINAL.IPYNB
# ==============================================================================
with tab_analisis:
    st.subheader("📖 Analisis Komprehensif PCD_FINAL.ipynb")
    st.markdown("""
    Notebook **`PCD_FINAL.ipynb`** adalah proyek akhir mata kuliah **Pengolahan Citra Digital (PCD)** 
    yang disusun oleh **Kelompok 9 (PCD UPN "Veteran" Jawa Timur)**. Notebook ini mendemonstrasikan 
    alur lengkap segmentasi citra dari tahapan dasar hingga teknik berbasis optimasi tingkat lanjut.
    """)

    with st.expander("📌 Sel 0: Otsu Thresholding & Background Removal", expanded=True):
        st.markdown("""
        - **Tujuan:** Menghilangkan background dari citra dan menghasilkan citra transparan berformat BGRA/RGBA.
        - **Langkah-langkah:**
          1. Konversi citra ke Grayscale.
          2. Gaussian Blur (5x5) untuk mengurangi derau frekuensi tinggi.
          3. Otsu Thresholding otomatis tanpa penentuan manual nilai ambang batas.
          4. Perhitungan nilai rata-rata tepi citra; jika lebih dari 127 (background putih/terang), mask diinversi otomatis.
          5. Operasi morfologi *Opening* (menghilangkan noise) dan *Closing* (menutup lubang).
          6. Connected Components Analysis untuk mengisolasi komponen terhubung dengan luas piksel terbesar.
          7. Flood Fill untuk mengisi lubang dalam objek (*hole filling*).
          8. Penghalusan tepi mask dengan Gaussian Blur agar potongan tidak bergerigi (*aliasing*).
          9. Penyisipan mask ke Alpha Channel (Channel ke-4) citra BGRA.
        """)

    with st.expander("📌 Sel 1: Edge-Based Segmentation (Canny Edge - 5 Tahapan Teori)", expanded=False):
        st.markdown(r"""
        - **Tujuan:** Mengimplementasikan 5 tahapan dasar teori Canny Edge Detection secara eksplisit.
        - **5 Tahapan Teori:**
          1. **Tahap 1 (Grayscale):** Reduksi dimensi warna 3 channel menjadi 1 channel intensitas keabuan.
          2. **Tahap 2 (Gaussian Blur):** Konvolusi kernel Gaussian $5 \times 5$ ($\sigma=1.4$) untuk mereduksi noise sensitif turunan.
          3. **Tahap 3 (Deteksi Gradien Sobel):** Menghitung turunan pertama arah horisontal ($G_x$) dan vertikal ($G_y$) menggunakan kernel Sobel 3x3, lalu menghitung Magnitudo Gradien:
             $$G = \sqrt{G_x^2 + G_y^2}$$
          4. **Tahap 4 (Non-Maximum Suppression / NMS):** Menekan piksel gradien yang bukan maksimum lokal sepanjang arah sudut normal tepi (menghasilkan tepi setebal 1 piksel).
          5. **Tahap 5 (Hysteresis Thresholding):** Penggunaan dua ambang batas ($T_{low}$ dan $T_{high}$) adaptif berbasis median citra ($\sigma = 0.33$) untuk mempertahankan kesinambungan tepi.
        """)

    with st.expander("📌 Sel 2: Region-Based Segmentation (Region Growing - 5 Tahapan Teori)", expanded=False):
        st.markdown(r"""
        - **Tujuan:** Mengelompokkan piksel tetangga yang memiliki kesamaan intensitas dari titik benih (*seed point*).
        - **5 Tahapan Teori:**
          1. **Tahap 1 (Karakteristik Piksel):** Konversi Grayscale dan Gaussian Blur ringan.
          2. **Tahap 2 (Titik Awal / Seed Point):** Menentukan koordinat awal $(x, y)$, secara default titik tengah citra $(W/2, H/2)$.
          3. **Tahap 3 & 4 (Pemeriksaan Tetangga & Penggabungan):** Algoritma antrean (Breadth-First Search) memeriksa 8 tetangga. Piksel $(x_n, y_n)$ digabungkan jika:
             $$|I(x_n, y_n) - I(x_{seed}, y_{seed})| \le \text{Threshold}$$
          4. **Tahap 5 (Segmentasi Wilayah):** Menghasilkan mask biner wilayah homogen yang telah tumbuh.
        """)

    with st.expander("📌 Sel 3: Clustering-Based Segmentation (K-Means - 7 Tahapan Teori)", expanded=False):
        st.markdown(r"""
        - **Tujuan:** Segmentasi citra berdasarkan pengelompokan warna di ruang vektor RGB menjadi $K$ kelompok.
        - **7 Tahapan Teori:**
          1. **Tahap 1:** Menentukan jumlah cluster $K$ (misal $K=3$).
          2. **Tahap 2:** Menentukan centroid awal menggunakan teknik K-Means++ pada sampel representatif piksel.
          3. **Tahap 3:** Menghitung jarak Euclidean setiap piksel ke semua centroid:
             $$d(p, c_i) = \|p - c_i\|_2$$
          4. **Tahap 4:** Mengelompokkan piksel ke cluster dengan jarak terdekat.
          5. **Tahap 5:** Memperbarui posisi centroid sebagai nilai rata-rata (*mean*) piksel di setiap cluster.
          6. **Tahap 6:** Pengulangan proses (iterasi) hingga kondisi konvergensi kriteria terminasi ($EPS$ atau $MaxIter$) terpenuhi.
          7. **Tahap 7:** Rekonstruksi citra berdasarkan warna representatif centroid dan perhitungan sebaran piksel.
        """)

    with st.expander("📌 Sel 4: Watershed Segmentation (8 Tahapan Teori)", expanded=False):
        st.markdown(r"""
        - **Tujuan:** Memisahkan objek-objek yang saling bersentuhan atau bertumpuk menggunakan konsep topografi.
        - **8 Tahapan Teori:**
          1. **Tahap 1:** Grayscale & Gaussian Blur.
          2. **Tahap 2:** Thresholding Otsu Inverted untuk memisahkan objek primer.
          3. **Tahap 3:** Morfologi Opening untuk membersihkan noise kecil.
          4. **Tahap 4:** Penentuan *Sure Background* (Dilasi) dan *Sure Foreground* (Distance Transform L2 $\ge 0.4 \times Max$).
          5. **Tahap 5:** Penentuan *Unknown Region* (Pengurangan Sure Background - Sure Foreground).
          6. **Tahap 6:** Pelabelan Marker unik untuk setiap komponen foreground dan penanda 0 untuk wilayah unknown.
          7. **Tahap 7:** Penjalaran air Watershed (`cv2.watershed`) hingga membangun bendungan (*dam line*) batas antar-objek (bernilai -1).
          8. **Tahap 8:** Visualisasi label wilayah unik dan overlay garis batas merah pada citra asli.
        """)

    with st.expander("📌 Sel 5: Graph Cut-Based Segmentation (GrabCut - 8 Tahapan Teori)", expanded=False):
        st.markdown("""
        - **Tujuan:** Segmentasi foreground-background tingkat lanjut menggunakan pemodelan probabilitas warna dan minimisasi energi Graph Cut.
        - **8 Tahapan Teori:**
          1. **Tahap 1:** Pembacaan citra 3-channel dan Preprocessing Blur ringan.
          2. **Tahap 2:** Menentukan kotak pembatas (*Bounding Box*) dengan margin 5% dari tepi.
          3. **Tahap 3:** Inisialisasi Mask 4 status (BGD, PR_BGD, PR_FGD, FGD).
          4. **Tahap 4:** Pemodelan warna Foreground dan Background menggunakan Gaussian Mixture Model (GMM).
          5. **Tahap 5:** Estimasi probabilitas piksel berdasarkan model GMM.
          6. **Tahap 6 & 7:** Optimasi energi min-cut/max-flow Graph Cut secara berulang (*iterative min-cut*).
          7. **Tahap 8:** Pembentukan mask biner akhir, citra transparan RGBA, dan garis kontur batas objek.
        """)


# ==============================================================================
# TAB 4: BATCH PROCESSING FOLDER
# ==============================================================================
with tab_batch:
    st.subheader("📁 Pemrosesan Citra Sekaligus (Batch Processing)")
    st.markdown("""
    Jalankan segmentasi otomatis pada seluruh berkas citra yang berada dalam sebuah direktori.
    """)

    folder_input_batch = st.text_input("Path Folder Citra Masukan:", value="sample_images")
    metode_batch = st.selectbox(
        "Pilih Metode Batch:",
        ["Otsu Thresholding", "Canny Edge", "Region Growing", "K-Means", "Watershed", "GrabCut", "Semua Metode"]
    )

    if st.button("▶️ Mulai Proses Batch", type="primary"):
        if not os.path.exists(folder_input_batch):
            st.error(f"Direktori '{folder_input_batch}' tidak ditemukan.")
        else:
            files_batch = [
                os.path.join(folder_input_batch, f) for f in os.listdir(folder_input_batch)
                if f.lower().endswith((".jpg", ".png", ".jfif", ".webp", ".jpeg"))
            ]
            if not files_batch:
                st.warning("Tidak ditemukan file citra valid di folder tersebut.")
            else:
                st.write(f"Ditemukan **{len(files_batch)} citra**. Memulai pemrosesan...")
                bar = st.progress(0)
                folder_out_batch = "hasil_batch_web"
                os.makedirs(folder_out_batch, exist_ok=True)

                for idx, fpath in enumerate(files_batch):
                    bgr = cv2.imread(fpath)
                    fname = os.path.splitext(os.path.basename(fpath))[0]
                    if bgr is not None:
                        if metode_batch in ["Otsu Thresholding", "Semua Metode"]:
                            r = proses_otsu_thresholding(bgr)
                            cv2.imwrite(os.path.join(folder_out_batch, f"{fname}_otsu_mask.png"), r["mask_final"])
                        if metode_batch in ["Canny Edge", "Semua Metode"]:
                            r = proses_canny_edge(bgr)
                            cv2.imwrite(os.path.join(folder_out_batch, f"{fname}_canny.png"), r["edges"])
                        if metode_batch in ["K-Means", "Semua Metode"]:
                            r = proses_kmeans_clustering(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), K=3)
                            cv2.imwrite(os.path.join(folder_out_batch, f"{fname}_kmeans.png"), cv2.cvtColor(r["segmented"], cv2.COLOR_RGB2BGR))
                        if metode_batch in ["Watershed", "Semua Metode"]:
                            r = proses_watershed_segmentation(bgr)
                            cv2.imwrite(os.path.join(folder_out_batch, f"{fname}_watershed.png"), cv2.cvtColor(r["hasil_overlay"], cv2.COLOR_RGB2BGR))
                        if metode_batch in ["GrabCut", "Semua Metode"]:
                            r = proses_grabcut_segmentation(bgr, iterasi=3)
                            cv2.imwrite(os.path.join(folder_out_batch, f"{fname}_grabcut.png"), cv2.cvtColor(r["hasil_segmentasi_rgb"], cv2.COLOR_RGB2BGR))

                    bar.progress((idx + 1) / len(files_batch))

                st.success(f"🎉 Pemrosesan batch selesai! Hasil tersimpan di folder `{folder_out_batch}`.")


# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 13px;'>
    Studio Pemrosesan Citra Digital (PCD) | Diadaptasi dari Proyek Akhir Kelompok 9 PCD UPN "Veteran" Jawa Timur
</div>
""", unsafe_allow_html=True)
