import cv2
import numpy as np
import pcd_segmentation as pcd

img_bgr = cv2.imread('sample_images/gajah.jpg')
assert img_bgr is not None, "Failed to load test image"
print("Test image loaded successfully:", img_bgr.shape)

# 1. Otsu
res_otsu = pcd.proses_otsu_thresholding(img_bgr)
print("1. Otsu OK. Threshold:", res_otsu["threshold_value"])

# 2. Canny
res_canny = pcd.proses_canny_edge(img_bgr)
print("2. Canny OK. Thresholds:", res_canny["threshold_bawah"], res_canny["threshold_atas"])

# 3. Region Growing
res_rg = pcd.proses_region_growing(img_bgr)
print("3. Region Growing OK. Grown pixels:", res_rg["pixel_count"])

# 4. K-Means
res_km = pcd.proses_kmeans_clustering(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), K=3)
print("4. K-Means OK. Compactness:", res_km["compactness"])

# 5. Watershed
res_ws = pcd.proses_watershed_segmentation(img_bgr)
print("5. Watershed OK. Objects detected:", res_ws["jumlah_objek"])

# 6. GrabCut
res_gc = pcd.proses_grabcut_segmentation(img_bgr, iterasi=3)
print("6. GrabCut OK. Mask shape:", res_gc["mask_final"].shape)

print("\n>>> ALL 6 SEGMENTATION METHODS TESTED AND PASSED SUCCESSFULLY! <<<")
