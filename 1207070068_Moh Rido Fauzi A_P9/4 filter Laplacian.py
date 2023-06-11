import cv2
import matplotlib.pyplot as plt

img0 = cv2.imread('istana.jpg')  # Membaca gambar grayscale.jpg

gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)  # Mengkonversi gambar menjadi citra grayscale

img = cv2.GaussianBlur(gray, (3, 3), 0)  # Menghilangkan noise dengan Gaussian Blur

laplacian = cv2.Laplacian(img, cv2.CV_64F)  # Melakukan operasi konvolusi dengan kernel Laplacian

cv2.imwrite('preprocessed_image.jpg', img)  # Menyimpan citra hasil preprocessing ke dalam file 'preprocessed_image.jpg'

cv2.imwrite('laplacian_image.jpg', laplacian)  # Menyimpan citra hasil operasi Laplacian ke dalam file 'laplacian_image.jpg'

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')  # Menampilkan citra hasil preprocessing pada subplot pertama
plt.title('Original')  # Memberi judul pada subplot pertama
plt.xticks([])  # Menghilangkan label sumbu x pada subplot pertama
plt.yticks([])  # Menghilangkan label sumbu y pada subplot pertama

plt.subplot(1, 2, 2)
plt.imshow(laplacian, cmap='gray')  # Menampilkan citra hasil operasi Laplacian pada subplot kedua
plt.title('Laplacian')  # Memberi judul pada subplot kedua
plt.xticks([])  # Menghilangkan label sumbu x pada subplot kedua
plt.yticks([])  # Menghilangkan label sumbu y pada subplot kedua

plt.show()  # Menampilkan plot dengan citra hasil preprocessing dan hasil Laplacian
