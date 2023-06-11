import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar
img = cv2.imread('istana.jpg', cv2.IMREAD_GRAYSCALE)  # Menggunakan gambar dengan format JPG

# Filter Prewitt
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

img_prewittx = cv2.filter2D(img, -1, kernelx)
img_prewitty = cv2.filter2D(img, -1, kernely)
img_prewitt = cv2.add(img_prewittx, img_prewitty)

fig, axes = plt.subplots(4, 2, figsize=(20, 20))
axes = axes.ravel()

axes[0].imshow(img, cmap='gray')
axes[0].set_title("Citra Input")
axes[1].hist(img.ravel(), bins=256)
axes[1].set_title("Histogram Citra Input")

axes[2].imshow(img_prewittx, cmap='gray')
axes[2].set_title("Citra Output Prewitt X")
axes[3].hist(img_prewittx.ravel(), bins=256)
axes[3].set_title("Histogram Citra Output Prewitt X")

axes[4].imshow(img_prewitty, cmap='gray')
axes[4].set_title("Citra Output Prewitt Y")
axes[5].hist(img_prewitty.ravel(), bins=256)
axes[5].set_title("Histogram Citra Output Prewitt Y")

axes[6].imshow(img_prewitt, cmap='gray')
axes[6].set_title("Citra Output Prewitt")
axes[7].hist(img_prewitt.ravel(), bins=256)
axes[7].set_title("Histogram Citra Output Prewitt")

fig.tight_layout()
plt.show()
