# 🔬 Studio Segmentasi Pengolahan Citra Digital (FP-PCD)
**Studi Komparasi & Visualisasi Interaktif 6 Metode Segmentasi Citra Digital**  
*Proyek Akhir Mata Kuliah Pengolahan Citra Digital (PCD) — Kelompok 9, UPN "Veteran" Jawa Timur*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-5.x-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

---

## 📌 Tentang Proyek
Repositori ini berisi implementasi komprehensif, modular, dan interaktif dari 6 algoritma segmentasi citra digital yang diadaptasi langsung dari berkas praktikum [`PCD_FINAL.ipynb`](./PCD_FINAL.ipynb). 

Proyek ini menyediakan dua mode antarmuka:
1. **🌐 Antarmuka Web Interaktif (Streamlit):** Memungkinkan pengguna memilih citra sampel atau mengunggah citra sendiri, menyetel parameter algoritma melalui *slider* secara langsung (*real-time*), melihat visualisasi setiap tahapan teori secara berdampingan, dan membandingkan performa seluruh metode secara bersamaan (*side-by-side*).
2. **💻 Skrip Antarmuka Terminal (CLI):** Untuk eksekusi otomatis atau pemrosesan batch massal pada berkas citra dalam sebuah direktori.

---

## 🌟 6 Metode Segmentasi yang Diimplementasikan

| No | Metode Segmentasi | Pendekatan | Tahapan Teori Ilmiah | Output Utama |
|---|---|---|---|---|
| 1 | **Otsu Thresholding & Background Removal** | Ambang Batas Global Adaptif | Memaksimalkan variansi antar-kelas ($\sigma_B^2$), deteksi tepi untuk auto-inversi, morfologi *opening/closing*, seleksi area terbesar, dan *flood fill hole filling*. | Mask biner bersih, Foreground transparan (BGRA/RGBA) |
| 2 | **Edge-Based Segmentation (Canny Edge)** | Turunan Spasial Gradien | **5 Tahap Teori:** 1. Grayscale $\rightarrow$ 2. Gaussian Blur ($\sigma=1.4$) $\rightarrow$ 3. Gradien Sobel $3 \times 3$ $\rightarrow$ 4. *Non-Maximum Suppression* (NMS) $\rightarrow$ 5. *Hysteresis Thresholding* adaptif. | Peta kontur/garis tepi biner 1 piksel |
| 3 | **Region-Based Segmentation (Region Growing)** | Kesamaan Spasial Wilayah | **5 Tahap Teori:** 1. Karakteristik intensitas $\rightarrow$ 2. Penentuan *Seed Point* awal $\rightarrow$ 3 & 4. Antrean BFS 8-tetangga ($|I - I_{seed}| \le \Delta T$) $\rightarrow$ 5. Mask wilayah homogen. | Wilayah segmen kontinu & highlight overlay |
| 4 | **Clustering-Based Segmentation (K-Means)** | Partisi Ruang Vektor Warna (RGB) | **7 Tahap Teori:** 1. Nilai $K$ $\rightarrow$ 2. Centroid awal *K-Means++* $\rightarrow$ 3. Jarak Euclidean $\rightarrow$ 4. Partisi klaster $\rightarrow$ 5. Update mean $\rightarrow$ 6. Konvergensi iterasi $\rightarrow$ 7. Rekonstruksi citra. | Citra terkuantisasi $K$-warna & palet distribusi |
| 5 | **Watershed Segmentation** | Morfologi Topografi Relief | **8 Tahap Teori:** 1. Blur $\rightarrow$ 2. Otsu Inverted $\rightarrow$ 3. Opening $\rightarrow$ 4. Sure FG (*Distance Transform*) & Sure BG (Dilasi) $\rightarrow$ 5. Unknown Region $\rightarrow$ 6. Marker $\rightarrow$ 7. Watershed flooding $\rightarrow$ 8. Batas relief (-1). | Garis pemisah objek bertumpuk & label pseudo-color |
| 6 | **Graph Cut Segmentation (GrabCut)** | Pemodelan GMM & Energi Graph Cut | **8 Tahap Teori:** 1. Preprocessing $\rightarrow$ 2. Bounding Box $\rightarrow$ 3. Inisialisasi Mask 4-state $\rightarrow$ 4. Pemodelan warna GMM $\rightarrow$ 5. Probabilitas piksel $\rightarrow$ 6 & 7. Iterative Min-Cut $\rightarrow$ 8. Segmentasi RGBA. | Ekstraksi objek alami kompleks & batas kontur |

---

## 📁 Struktur Direktori Repositori

```text
FP-PCD/
├── PCD_FINAL.ipynb         # Notebook Jupyter asli (Google Colab)
├── pcd_segmentation.py     # Pustaka inti (Core Library) 6 metode segmentasi
├── app.py                  # Aplikasi antarmuka web interaktif (Streamlit)
├── main.py                 # Skrip antarmuka terminal (CLI) untuk pemrosesan batch
├── test_segmentation.py    # Skrip unit testing & verifikasi pipeline algoritma
├── requirements.txt        # Daftar dependensi paket Python
├── .gitignore              # Konfigurasi pengabaian file Git (venv, cache, outputs)
├── README.md               # Dokumentasi lengkap proyek
└── sample_images/          # Citra sampel asli dari dataset notebook
    ├── gajah.jpg
    ├── stroberi.jpg
    ├── kucing.jpg
    ├── telur.jpg
    ├── Orange.jpg
    ├── kelinci.jpg
    └── ayam.jpg
```

---

## 🚀 Panduan Setup & Instalasi

Ikuti langkah-langkah berikut untuk menjalankan proyek di komputer lokal Anda:

### 1. Kloning Repositori
```bash
git clone https://github.com/<username-anda>/<nama-repo>.git
cd <nama-repo>
```

### 2. Buat & Aktifkan Virtual Environment (`venv`)

- **Di Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```

- **Di Windows (Command Prompt):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate.bat
  ```

- **Di macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Pasang Dependensi
Pasang seluruh pustaka yang dibutuhkan melalui `requirements.txt`:
```bash
pip install -r requirements.txt
```

> **Tips:** Jika proses instalasi di Windows mengalami *timeout* saat mengunduh `pyarrow` atau `streamlit`, tambahkan opsi timeout dan mirror alternatif:
> ```bash
> pip install -r requirements.txt --default-timeout=1000 -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

## 💻 Cara Menjalankan Proyek

### A. Menjalankan Website Interface (Streamlit)
Pastikan virtual environment telah aktif, lalu jalankan:
```bash
streamlit run app.py
```
*(Atau gunakan perintah: `python -m streamlit run app.py`)*

Aplikasi akan otomatis terbuka di browser pada alamat:
👉 **`http://localhost:8501`**

#### Fitur-Fitur Website:
- **🧪 Eksplorasi Tahapan Metode:** Eksplorasi langkah demi langkah dari setiap algoritma dengan slider parameter interaktif.
- **⚖️ Komparasi 6 Metode Sekaligus:** Menjalankan ke-6 metode pada citra yang sama secara bersamaan lengkap dengan tabel analisis waktu komputasi (ms) dan performa.
- **📥 Ekspor Hasil:** Unduh hasil segmentasi (mask biner dan citra transparan PNG) langsung dari website.
- **📁 Pemrosesan Batch:** Proses seluruh folder citra sekaligus hanya dengan satu kali klik.

---

### B. Menjalankan Versi Skrip Terminal (CLI)
Jika ingin memproses citra langsung melalui terminal tanpa membuka browser:

```bash
# Memproses seluruh gambar di folder sample_images untuk semua 6 metode:
python main.py

# Memproses satu gambar spesifik dengan metode tertentu:
python main.py -i sample_images/gajah.jpg -m otsu
python main.py -i sample_images/kucing.jpg -m canny
python main.py -i sample_images/telur.jpg -m kmeans --k 3
python main.py -i sample_images/ayam.jpg -m watershed
python main.py -i sample_images/Orange.jpg -m grabcut --iterasi 5
```

Hasil segmentasi dan plot matplotlib akan otomatis tersimpan pada folder keluaran:
`hasil_otsu/`, `hasil_edge_5_tahap/`, `hasil_region_growing/`, `hasil_kmeans/`, `hasil_watershed/`, dan `hasil_grabcut/`.

---

### C. Menjalankan Uji Verifikasi (Unit Test)
Untuk memastikan seluruh fungsi segmentasi berjalan 100% tanpa kendala di lingkungan Anda:
```bash
python test_segmentation.py
```

---

## 👥 Tim Penyusun
Proyek ini dikembangkan oleh **Kelompok 9** pada mata kuliah **Pengolahan Citra Digital (PCD)**, Program Studi Informatika, UPN "Veteran" Jawa Timur.

---

## 📄 Lisensi
Didistribusikan di bawah lisensi MIT. Silakan gunakan dan kembangkan untuk keperluan akademik dan pembelajaran.
