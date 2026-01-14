import cv2
import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple

CV2Img = NDArray[np.uint8]


def softmax(x: NDArray[np.float32], temp: float = 1.0, axis: int = -1):
    """
    Softmax with temperature.

    Args:
        x: Input array.
        temp: Temprature (default: 1.0).
        axis: Dimension along which softmax is computed (default: -1).
    """
    z = x - np.max(x, axis=axis, keepdims=True)
    z *= 1.0 / temp
    np.exp(z, out=z)
    z /= np.sum(z, axis=axis, keepdims=True)
    return z


def multi_hist(
    matrices: NDArray[np.float32],
    ix_mask: tuple[list[int], list[int]],
    bins: NDArray[np.float32],
    drop_diag: bool = False,
) -> NDArray[np.uint8]:
    """
    Computes N histograms, one per row given an (N x D) matrix.

    Args:
        matrices: An (N x D) numpy array containing N sets of values.
        ix_mask: Indices in the matrices to use when computing the histogram.
        bins: Array with shape (B,) corresponding to histogram bin edges.
    Returns:
        An (N x B) array containing N histograms. B is the number of bins (default: 1000).
    """
    n_slices = matrices.shape[0]
    num_bins = len(bins) - 1
    masked_data = matrices[:, ix_mask[0], ix_mask[1]]
    if drop_diag:
        masked_data = masked_data[:, ~np.eye(masked_data.shape[1], dtype=bool)]
    bin_indices = np.searchsorted(bins, masked_data, side="right") - 1
    np.clip(bin_indices, 0, num_bins - 1, out=bin_indices)
    offsets = np.arange(n_slices).reshape(-1, 1, 1) * num_bins
    global_indices = (bin_indices + offsets).ravel()
    return (
        np.bincount(global_indices, minlength=n_slices * num_bins)
        .reshape(n_slices, num_bins)
        .astype(np.uint8)
    )


def multi_otsu(
    hist: NDArray[np.float32], bins: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Computes thresholds for several histograms using the Otsu method.

    Args:
        hist: An (N, B) numpy array containing N histograms.
        bins: Array with shape (B,) corresponding to histogram bin edges.
    Returns:
        thresholds: An (N,) shaped numpy array with Otsu thresholds, one per histogram.
        eta: The Otsu effectiveness metric = max inter-class variance / global variance.
    """
    # Compute bin_centers (shape = (B,))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # Normalize histograms to probabilities (shape = (N, B))
    prob = hist.astype(np.float32) / hist.sum(axis=1, keepdims=True)
    # Compute cumulative sums across the bins (axis=1)
    omega = np.cumsum(prob, axis=1)  # shape: (N, BINS)
    mu = np.cumsum(prob * bin_centers, axis=1)  # shape: (N, BINS)
    mu_t = mu[:, -1:]  # Total mean for each row: (D, 1)
    # Compute inter-class variance for all thresholds
    numerator = (mu_t * omega - mu) ** 2
    denominator = omega * (1 - omega)
    # Compute inter-class variance
    sigma_b_squared = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=(denominator > 1e-9),
    )
    # Compute global variance
    sigma_g_squared = np.sum(prob * (bin_centers - mu_t) ** 2, axis=1)
    # Thresholds = indices of max inter-class variance
    max_indices = np.argmax(sigma_b_squared, axis=1)
    thresholds = bin_centers[max_indices]
    max_sigma_b_squared = sigma_b_squared[range(len(hist)), max_indices]
    # Otsu effectiveness metric = sigma_b_squared / sigma_g_squared
    eta = np.divide(
        max_sigma_b_squared,
        sigma_g_squared,
        out=np.zeros_like(sigma_g_squared),
        where=(sigma_g_squared > 1e-9),
    )
    return thresholds, eta


class ImagePreprocTransforms(NamedTuple):
    size: tuple[int, int]
    col: str = "gray"  # ["bgr", "gray", "hsv"]
    edges: bool = False  # if true, use Canny edge
    log_polar: bool = False  # if true, use log polar form


class MemoizedImage:
    """
    Wrapper around an cv2 image that caches transformations applied it.
    Useful if you need to apply the same or similar transforms on the image several times.
    """

    def __init__(self, img):
        self.col = "gray" if img.ndim == 2 else "bgr"
        self.edges = False
        self.log_polar = False
        orig_transform = ImagePreprocTransforms(
            img.shape[:2], self.col, self.edges, self.log_polar
        )
        self.cache = {orig_transform: img}
        self.orig_area = img.shape[0] * img.shape[1]

    def preproc(self, transforms: ImagePreprocTransforms) -> NDArray[np.uint8]:
        #  Simply recall if already in cache
        if transforms in self.cache:
            return self.cache[transforms]
        # Find the image in cache with the smallest size bigger than the target size
        img_area = transforms.size[0] * transforms.size[1]
        min_size_key = min(
            filter(
                lambda key: (
                    key[0][0] * key[0][1] >= img_area
                    or key[0][0] * key[0][1] == self.orig_area
                )
                and key[1] == self.col,
                self.cache.keys(),
            )
        )
        img = self.cache[min_size_key]
        # Keep only edges
        if transforms.edges:
            img = cv2.Canny(img, 100, 200)
            if self.col != "gray":
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Apply log polar transform
        if transforms.log_polar:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            img = cv2.warpPolar(
                img,
                dsize=(w, h),
                center=center,
                maxRadius=min(center),
                flags=cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR,
            )
        # Resize
        if transforms.size != img.shape[:2]:
            img = cv2.resize(img, transforms.size[::-1], interpolation=cv2.INTER_AREA)
        # Cache the resized version in the _default_ color space
        self.cache[(transforms.size, self.col, self.edges, self.log_polar)] = img
        # Convert to the target color space
        if transforms.col == "gray":
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif transforms.col == "hsv":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Update cache
        if transforms not in self.cache:
            self.cache[transforms] = img
        return img
