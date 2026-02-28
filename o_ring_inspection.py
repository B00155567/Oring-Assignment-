import numpy as np
import cv2
import os
import glob

# Histogram + Otsu

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
        var = cum_pix * rem_pix * (
            (cum_sum / cum_pix) -
            (total_sum - cum_sum) / rem_pix
        ) ** 2

        if var > max_var:
            max_var, best_t = var, t

    return best_t

#Morphological Closing

def _morph(img, k, erode_mode):
    h, w = img.shape
    pad = k // 2

    padded = np.pad(img, pad, constant_values=0)
    out = np.full_like(img, 255 if erode_mode else 0)

    for di in range(-pad, pad + 1):
        for dj in range(-pad, pad + 1):
            win = padded[pad + di:pad + di + h,
                        pad + dj:pad + dj + w]

            if erode_mode:
                out = np.minimum(out, win)
            else:
                out = np.maximum(out, win)

    return out


def closing(img, k=3):
    # dilation followed by erosion
    return _morph(_morph(img, k, False), k, True)

# Connected Components

class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        if x not in self.p:
            self.p[x] = x
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.p[px] = py


def connected_components(binary):
    h, w = binary.shape
    labelled = np.zeros((h, w), dtype=np.int32)

    uf = UnionFind()
    label_map = {}
    next_lab = 1

    for i in range(h):
        for j in range(w):
            if binary[i, j] == 255:

                # check north and west neighbors
                neighbors = [(i - 1, j), (i, j - 1)]
                neighbors = [
                    (a, b) for a, b in neighbors
                    if 0 <= a < h and 0 <= b < w
                    and binary[a, b] == 255
                ]

                if not neighbors:
                    label_map[(i, j)] = next_lab
                    next_lab += 1
                else:
                    min_label = min(label_map[n] for n in neighbors)
                    label_map[(i, j)] = min_label

                    for n in neighbors:
                        uf.union(label_map[n], min_label)

    roots = {}
    for (i, j), lab in label_map.items():
        r = uf.find(lab)
        if r not in roots:
            roots[r] = len(roots) + 1
        labelled[i, j] = roots[r]

    regions = {
        L: list(zip(*np.where(labelled == L)))
        for L in np.unique(labelled) if L > 0
    }

    return labelled, regions

# Processing Pipeline (So Far)

def process_image(path):
    img = cv2.imread(path)
    if img is None:
        return

    # Convert to grayscale manually
    gray = (
        0.299 * img[:, :, 2] +
        0.587 * img[:, :, 1] +
        0.114 * img[:, :, 0]
    ).astype(np.uint8)

    # Otsu threshold
    hist = compute_histogram(gray)
    thresh = otsu_threshold(hist)

    binary = np.where(gray >= thresh, 255, 0).astype(np.uint8)

    # Morphological closing
    binary = closing(binary)

    # Connected components
    labelled, regions = connected_components(binary)

    # Extract largest region
    mask = np.zeros_like(binary)
    if regions:
        main_region = max(regions.values(), key=len)
        ys = np.array([p[0] for p in main_region])
        xs = np.array([p[1] for p in main_region])
        mask[ys, xs] = 255

    cv2.imwrite("mask_" + os.path.basename(path), mask)


if __name__ == "__main__":
    for p in glob.glob("Orings/*.jpg"):
        process_image(p)