import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar
img = cv2.imread('istana.jpg', cv2.IMREAD_GRAYSCALE)  # Membaca gambar 'istana.jpg' dalam mode grayscale

# Canny Edge Detection
img_canny = cv2.Canny(img, 100, 200)  # Menggunakan metode Canny untuk mendeteksi tepi pada citra

# Menampilkan citra dan histogram
fig, axes = plt.subplots(2, 2, figsize=(20, 20))  # Membuat subplot dengan 2 baris dan 2 kolom
ax = axes.ravel()  # Mengubah array subplot menjadi array 1 dimensi

ax[0].imshow(img, cmap='gray')  # Menampilkan citra asli pada subplot pertama
ax[0].set_title("Citra Input")  # Memberi judul pada subplot pertama
ax[1].hist(img.ravel(), bins=256)  # Menampilkan histogram citra asli pada subplot kedua
ax[1].set_title("Histogram Citra Input")  # Memberi judul pada subplot kedua

ax[2].imshow(img_canny, cmap='gray')  # Menampilkan citra hasil Canny pada subplot ketiga
ax[2].set_title("Citra Output Canny")  # Memberi judul pada subplot ketiga
ax[3].hist(img_canny.ravel(), bins=256)  # Menampilkan histogram citra hasil Canny pada subplot keempat
ax[3].set_title("Histogram Citra Output Canny")  # Memberi judul pada subplot keempat

fig.tight_layout()  # Menyusun subplot agar terlihat rapi
plt.show()  # Menampilkan semua subplot
