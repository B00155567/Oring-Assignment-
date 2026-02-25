import numpy as np
import cv2
import os
import glob

def compute_histogram(gray):
    return np.bincount(gray.ravel(), minlength=256)

def otsu_threshold(hist):
    total = np.sum(hist)
    if total == 0:
        return 128

    total_sum = np.sum(np.arange(256) * hist)
    best_t, max_var = 0, 0.0
    cum_sum, cum_pix = 0, 0

    for t in range(256):
        cum_pix += hist[t]
        if cum_pix == 0:
            continue
        rem_pix = total - cum_pix
        if rem_pix == 0:
            break

        cum_sum += t * hist[t]
        var = cum_pix * rem_pix * ((cum_sum / cum_pix) -
               (total_sum - cum_sum) / rem_pix) ** 2

        if var > max_var:
            max_var, best_t = var, t

    return best_t


def process_image(path):
    img = cv2.imread(path)
    if img is None:
        return

    gray = (0.299 * img[:, :, 2] +
            0.587 * img[:, :, 1] +
            0.114 * img[:, :, 0]).astype(np.uint8)

    hist = compute_histogram(gray)
    thresh = otsu_threshold(hist)

    binary = np.where(gray >= thresh, 255, 0).astype(np.uint8)
    cv2.imwrite("binary_" + os.path.basename(path), binary)


if __name__ == "__main__":
    for p in glob.glob("Orings/*.jpg"):
        process_image(p)