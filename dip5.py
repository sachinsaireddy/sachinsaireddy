import cv2
import numpy as np
from matplotlib import pyplot as plt
def histogram_equalization(img):
# Calculate the histogram of the input image
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
# Calculate the cumulative distribution function of the histogram
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()
    cdf_equalized = np.round(cdf_normalized * 255)
    img_equalized = cdf_equalized[img]
# Convert the data type of the image to uint8
    img_equalized = np.uint8(img_equalized)
    return img_equalized
def histogram_matching(source_img, reference_img):
    source_hist, _ = np.histogram(source_img.flatten(), 256, [0, 256])
    reference_hist, _ = np.histogram(reference_img.flatten(), 256, [0, 256])
    source_cdf = source_hist.cumsum()
    reference_cdf = reference_hist.cumsum()
# Normalize the CDFs to have values between 0 and 1
    source_cdf_normalized = source_cdf / source_cdf.max()
    reference_cdf_normalized = reference_cdf / reference_cdf.max()
# Calculate the mapping function from the source to the reference histogram
    mapping_function = np.interp(source_cdf_normalized,
    reference_cdf_normalized, range(256))
# Apply the mapping function to the source image
    matched_img = np.round(np.interp(source_img.flatten(), range(256),
    mapping_function))
    matched_img = matched_img.reshape(source_img.shape)
# Convert the data type of the matched image to uint8
    matched_img = np.uint8(matched_img)
    return matched_img
#Taking the image
img1 = cv2.imread('pout-dark.jpg')
img2 = cv2.imread('pout-bright.jpg')
plt.imshow(img1)
plt.title("Original Image ")
plt.show()

hist_img = histogram_equalization(img1)
plt.imshow(hist_img)
plt.title("QN_1)Histogram_equalized Image ")
plt.show()


