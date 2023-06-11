import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar
img = cv2.imread('dog.jpg', cv2.IMREAD_GRAYSCALE)  # Membaca gambar 'dog.jpg' dalam mode grayscale

# Mengaplikasikan filter Sobel
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # Menggunakan filter Sobel pada sumbu x
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # Menggunakan filter Sobel pada sumbu y
sobel = np.abs(sobelx) + np.abs(sobely)  # Menggabungkan hasil filter Sobel sumbu x dan sumbu y

fig, axes = plt.subplots(4, 2, figsize=(20, 20))  # Membuat subplot dengan 4 baris dan 2 kolom
axes = axes.ravel()  # Membentuk array 1 dimensi dari subplot

axes[0].imshow(img, cmap='gray')  # Menampilkan gambar asli pada subplot pertama
axes[0].set_title("Citra Input")  # Memberi judul pada subplot pertama
axes[1].hist(img.ravel(), bins=256)  # Menampilkan histogram gambar asli pada subplot kedua
axes[1].set_title('Histogram Input')  # Memberi judul pada subplot kedua

axes[2].imshow(sobelx, cmap='gray')  # Menampilkan hasil filter Sobel sumbu x pada subplot ketiga
axes[2].set_title("Citra Output Sobel X")  # Memberi judul pada subplot ketiga
axes[3].hist(sobelx.ravel(), bins=256)  # Menampilkan histogram hasil filter Sobel sumbu x pada subplot keempat
axes[3].set_title('Histogram Citra Output Sobel X')  # Memberi judul pada subplot keempat

axes[4].imshow(sobely, cmap='gray')  # Menampilkan hasil filter Sobel sumbu y pada subplot kelima
axes[4].set_title("Citra Output Sobel Y")  # Memberi judul pada subplot kelima
axes[5].hist(sobely.ravel(), bins=256)  # Menampilkan histogram hasil filter Sobel sumbu y pada subplot keenam
axes[5].set_title('Histogram Citra Output Sobel Y')  # Memberi judul pada subplot keenam

axes[6].imshow(sobel, cmap='gray')  # Menampilkan hasil filter Sobel pada subplot ketujuh
axes[6].set_title("Citra Output Sobel")  # Memberi judul pada subplot ketujuh
axes[7].hist(sobel.ravel(), bins=256)  # Menampilkan histogram hasil filter Sobel pada subplot kedelapan
axes[7].set_title('Histogram Citra Output Sobel')  # Memberi judul pada subplot kedelapan

plt.tight_layout()  # Menyusun subplot agar terlihat rapi
plt.show()  # Menampilkan semua subplot
