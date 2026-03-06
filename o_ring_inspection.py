import numpy as np
import cv2, os, time, glob

# minimum size allowed for the main ring region
# FRAGMENT_RATIO is used to decide if smaller blobs count as broken pieces
MIN_AREA, FRAGMENT_RATIO = 100, 0.02

# CIRC_MIN = lowest circularity allowed before the ring is considered irregular
# EXTENT_MAX = used to detect C-shaped rings (too filled bounding box)
CIRC_MIN, EXTENT_MAX = 0.12, 0.80

# solidity checks used to detect tears or notches in the ring
# values around these thresholds indicate abnormal shape
SOL_MID_LOW, SOL_MID_HIGH, SOL_HIGH = 0.744, 0.752, 0.753

# perimeter ratio detects rough edges
# CIRC_EDGE prevents this rule triggering on non-circular shapes
PR_EDGE, CIRC_EDGE = 7.40, 0.20


# Build a 256-bin histogram of pixel intensities for the grayscale image.
def compute_histogram(gray):
    return np.bincount(gray.ravel(), minlength=256)


# Compute the optimal binary threshold using Otsu's method to separate foreground from background.
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

        var = cum_pix * rem_pix * ((cum_sum / cum_pix) - (total_sum - cum_sum) / rem_pix) ** 2

        if var > max_var:
            max_var, best_t = var, t

    return best_t


# Apply erosion or dilation with a k×k structuring element (internal helper).
def _morph(img, k, erode_mode):
    h, w, pad = img.shape[0], img.shape[1], k // 2
    padded = np.pad(img, pad, constant_values=0)
    out = np.full_like(img, 255 if erode_mode else 0)

    for di in range(-pad, pad + 1):
        for dj in range(-pad, pad + 1):

            win = padded[pad + di : pad + di + h, pad + dj : pad + dj + w]

            out = np.minimum(out, win) if erode_mode else np.maximum(out, win)

    return out


# Morphological closing (dilation followed by erosion) to fill small gaps in the binary image.
def closing(img, k=3):
    return _morph(_morph(img, k, False), k, True)


# Union-Find data structure for grouping connected pixels into components.
class UnionFind:

    # Create an empty Union-Find structure.
    def __init__(self):
        self.p = {}

    # Return the root representative of the set containing x.
    def find(self, x):
        if x not in self.p:
            self.p[x] = x

        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])

        return self.p[x]

    # Merge the sets containing x and y.
    def union(self, x, y):
        px, py = self.find(x), self.find(y)

        if px != py:
            self.p[px] = py


# Label all connected white regions in a binary image and return a label map plus a dict of pixel lists per region.
def connected_components(binary):
    h, w = binary.shape
    labelled = np.zeros((h, w), dtype=np.int32)

    uf, label_map, next_lab = UnionFind(), {}, 1

    for i in range(h):
        for j in range(w):

            if binary[i, j] == 255:

                nbrs = [(i - 1, j), (i, j - 1)]

                nbrs = [(a, b) for a, b in nbrs if 0 <= a < h and 0 <= b < w and binary[a, b] == 255]

                if not nbrs:
                    label_map[(i, j)] = next_lab
                    next_lab += 1

                else:
                    mlab = min(label_map[n] for n in nbrs)
                    label_map[(i, j)] = mlab

                    for n in nbrs:
                        uf.union(label_map[n], mlab)

    roots = {}

    for (i, j), lab in label_map.items():

        r = uf.find(lab)

        if r not in roots:
            roots[r] = len(roots) + 1

        labelled[i, j] = roots[r]

    regions = {L: list(zip(*np.where(labelled == L))) for L in np.unique(labelled) if L > 0}

    return labelled, regions


# Count the number of enclosed holes (background regions) inside the white foreground of the mask.
def count_holes(mask):
    pad = np.pad(mask, 1, constant_values=0)
    inv = 255 - pad

    inv[0, :] = inv[-1, :] = inv[:, 0] = inv[:, -1] = 0

    _, regions = connected_components(inv)

    H, W = inv.shape

    border = {L for L, pix in regions.items() for i, j in pix if i in (1, H-2) or j in (1, W-2)}

    return max(0, len(regions) - len(border))


# Compute the 4-connected perimeter length of a region (internal helper).
def _perim(pixels):
    if len(pixels) < 10:
        return 0

    ps = set(pixels)

    return sum(1 for i, j in pixels
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]
            if (i+di,j+dj) not in ps)


# Measure how round the region is (1 = perfect circle, lower = more irregular).
def circularity(pixels):
    p = _perim(pixels)

    return 4 * np.pi * len(pixels) / (p ** 2) if p else 0


# Perimeter divided by sqrt(area); high values indicate rough or jagged edges.
def perimeter_ratio(pixels):
    p = _perim(pixels)

    return p / np.sqrt(len(pixels)) if p else 0


# Ratio of area to bounding box area; high values suggest a C-shaped or filled ring.
def extent(pixels):
    if len(pixels) < 5:
        return 0

    ys, xs = [p[0] for p in pixels], [p[1] for p in pixels]

    return len(pixels) / ((max(ys) - min(ys) + 1) * (max(xs) - min(xs) + 1))


def convex_hull_area(pixels):
    """Compute the area of the convex hull enclosing the region (used for solidity)."""
    if len(pixels) < 3:
        return 0

    pset = set(pixels)

    boundary = [p for p in pixels if any((p[0]+di,p[1]+dj) not in pset for di,dj in [(-1,0),(1,0),(0,-1),(0,1)])]

    ps = [(p[1], p[0]) for p in (boundary if len(boundary) >= 3 else pixels)]

    cross = lambda o, a, b: (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    start = min(ps, key=lambda p: (p[1], p[0]))

    pts = sorted(ps, key=lambda p:
        (np.arctan2(p[1]-start[1], p[0]-start[0]) if p != start else -np.pi,
        (p[0]-start[0])**2 + (p[1]-start[1])**2))

    hull = []

    for p in pts:
        while len(hull) > 1 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    if len(hull) < 3:
        return 0

    return abs(sum(hull[i][0]*hull[(i+1)%len(hull)][1] -
                hull[(i+1)%len(hull)][0]*hull[i][1]
                for i in range(len(hull)))) / 2


# Area divided by convex hull area; dips indicate concavities such as tears or notches.
def solidity(pixels):
    if len(pixels) < 10:
        return 1

    h = convex_hull_area(pixels)

    return len(pixels) / h if h > 0 else 1


# Check if the O-ring is defective based on holes, shape metrics, fragments, and edge quality. Returns (defective, reason).
def is_defective(mask, regions):
    if not regions:
        return True, "No ring found"

    main_pix = max(regions.values(), key=len)
    main_area = len(main_pix)

    if main_area < MIN_AREA:
        return True, "Ring too small"

    # Fragment check: smaller blobs indicate broken pieces
    for L, pix in regions.items():
        if len(pix) < main_area * FRAGMENT_RATIO and len(pix) >= MIN_AREA:
            return True, "Fragments detected"

    # Hole count
    holes = count_holes(mask)
    if holes != 1:
        return True, f"Wrong hole count ({holes})"

    circ = circularity(main_pix)
    if circ < CIRC_MIN:
        return True, f"Low circularity ({circ:.3f})"

    ext = extent(main_pix)
    if ext > EXTENT_MAX:
        return True, f"C-shaped ({ext:.3f})"

    sol = solidity(main_pix)
    if SOL_MID_LOW <= sol <= SOL_MID_HIGH:
        return True, f"Tear/notch (solidity {sol:.3f})"
    if sol < SOL_HIGH and circ >= CIRC_EDGE:
        return True, f"Abnormal solidity ({sol:.3f})"

    pr = perimeter_ratio(main_pix)
    if pr > PR_EDGE and circ >= CIRC_EDGE:
        return True, f"Rough edges (pr={pr:.2f})"

    return False, "OK"


def process(img_path):
    """Load an image, segment the O-ring, run defect checks, and return (result, reason, elapsed_ms)."""
    img = cv2.imread(img_path)

    if img is None:
        return None, "Load failed", 0

    gray = (0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]).astype(np.uint8)

    t0 = time.perf_counter()

    hist = compute_histogram(gray)
    thresh = otsu_threshold(hist)

    binary = np.where(gray >= thresh, 255, 0).astype(np.uint8)

    binary = closing(binary)

    _, regions = connected_components(binary)

    mask = np.zeros_like(binary)

    if regions:
        main_pix = max(regions.values(), key=len)

        mask[np.array([p[0] for p in main_pix]),
            np.array([p[1] for p in main_pix])] = 255

    defective, reason = is_defective(mask, regions)

    result = "FAIL" if defective else "PASS"

    elapsed = (time.perf_counter() - t0) * 1000

    return result, reason, elapsed


# Run O-ring inspection on all JPG images in the Orings folder and print results.
def main():
    base = os.path.dirname(os.path.abspath(__file__))
    paths = sorted(glob.glob(os.path.join(base, "Orings", "*.jpg")))

    if not paths:
        print("No images in Orings/")
        return

    print("O-ring Inspection")
    print("-" * 50)

    for p in paths:

        result, reason, ms = process(p)

        print(f"{os.path.basename(p)}: {result} ({reason}) [{ms:.1f} ms]")

    print("-" * 50)


if __name__ == "__main__":
    main()