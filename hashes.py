from abc import ABC, abstractmethod
import cv2
import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort

from utils import MemoizedImage


class ImageHash(ABC):
    _THRESH_FUNCS = {"mean": np.mean, "median": np.median}

    def __init__(
        self,
        img_size: tuple[int, int],
        col: str,
        thresh: str = "mean",
        edges: bool = False,
    ):
        self.col = col
        self.img_size = img_size
        self.img_area = img_size[0] * img_size[1]
        self.thresh = thresh
        self.edges = edges
        self.thresh_func = self._THRESH_FUNCS.get(thresh)

    def preproc(self, img: MemoizedImage):
        # Resize, set color space and cast to fp32
        return img.preproc(self.img_size, self.col, self.edges).astype(np.float32)

    def bitvec(self, x: NDArray[np.float32]) -> NDArray[np.uint8]:
        # Threshold features and binarize to bit vectors
        T = self.thresh_func(x)
        return (x > T).astype(np.uint8).flatten()

    @abstractmethod
    def feat(self, img: MemoizedImage) -> NDArray[np.uint8]:
        # Must extract a perceptual hash (a binary feature vector) of an image
        pass

    def sim(self, x1: NDArray[np.uint8], x2: NDArray[np.uint8]) -> float:
        # Compute similarity between individual bit vectors (= 1 - Hamming distance) in [0, 1]
        return (x1 == x2).sum() / x1.size

    def sim_batch(self, x: NDArray[np.uint8], out=None) -> NDArray[np.float32]:
        # Compute similarity matrix given a stack of bit vectors
        if x.ndim == 1:
            x = x[np.newaxis, :]
        N, D = x.shape  # num samples, hash dimension
        if out is None:
            out = np.empty((N, N), dtype=np.float32)
        np.sum(
            x[:, np.newaxis, :] == x[np.newaxis, :, :],
            axis=2,
            dtype=np.float32,
            out=out,
        )
        np.divide(out, D, out=out)
        return out


class ColorHistHash(ImageHash):
    def __init__(
        self,
        img_size: int,
        channels: list[int],
        bins: list[int],
        ranges: list[int],
        col: str = "hsv",
        thresh: str = "mean",
    ):
        super().__init__((img_size, img_size), col, thresh)
        self.channels = channels
        self.bins = bins
        self.ranges = ranges

    def feat(self, img: MemoizedImage) -> NDArray[np.uint8]:
        img = self.preproc(img)
        h = cv2.calcHist([img], self.channels, None, self.bins, self.ranges)
        return self.bitvec(h)


class GaborHash(ImageHash):
    def __init__(
        self, img_size: int, model_path: str, thresh: str, edges: bool = False
    ):
        super().__init__((img_size, img_size), "gray", thresh, edges)
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def feat(self, img: MemoizedImage) -> NDArray[np.uint8]:
        img = self.preproc(img)
        img -= img.mean()
        img /= img.std() + 1e-8
        img = img[None, None, :, :]
        f = self.sess.run(None, {self.input_name: img})[0][0]
        return self.bitvec(f)


class SqueezeNetHash(ImageHash):
    def __init__(self, img_size: tuple[int, int], model_path: str, thresh: str):
        super().__init__((img_size, img_size), "col", thresh)
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def feat(self, img: MemoizedImage) -> NDArray[np.uint8]:
        img = self.preproc(img)
        img = img.transpose(2, 0, 1)[None, :, :, :]
        f = self.sess.run(None, {self.input_name: img})[0][0]
        return self.bitvec(f)


class PerceptualHash(ImageHash):
    def __init__(
        self, hash_size: int, highfreq_factor: int, thresh: str, edges: bool = False
    ):
        super().__init__(
            (hash_size * highfreq_factor, hash_size * highfreq_factor),
            "gray",
            thresh,
            edges,
        )
        self.hash_size = hash_size

    def feat(self, img: MemoizedImage) -> NDArray[np.uint8]:
        img = self.preproc(img)
        dct_low_freq = cv2.dct(img)[: self.hash_size, : self.hash_size]
        return self.bitvec(dct_low_freq)


class PixelHash(ImageHash):
    def __init__(self, hash_size: int, thresh: str, edges: bool = False):
        super().__init__((hash_size, hash_size), "gray", thresh, edges)

    def feat(self, img: MemoizedImage) -> NDArray[np.uint8]:
        img = self.preproc(img)
        return self.bitvec(img)


class WaveletHash(ImageHash):
    def __init__(
        self,
        hash_size: int,
        scale: int,
        thresh: str,
        blur: int = 0,
        edges: bool = False,
    ):
        super().__init__((hash_size * scale, hash_size * scale), "gray", thresh, edges)
        self.levels = int(np.log2(scale))
        self.blur = blur

    def _ensure_even_dims(self, img):
        """Crops 1 pixel if odd to make dims even."""
        h, w = img.shape[:2]
        new_h = h - (h % 2)
        new_w = w - (w % 2)
        if new_h != h or new_w != w:
            img = img[:new_h, :new_w]
        return img

    def _haar_single_level(self, img):
        """
        img: 2D float32 array with even dims.
        returns LL (2D float32) — low-low after one separable Haar step.
        """
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        # Horizontal (pair columns)
        a = img[:, 0::2]
        b = img[:, 1::2]
        # a and b shapes match because width is even
        Lh = (a + b) * inv_sqrt2
        # (Hh = (a - b) * inv_sqrt2)  # we don't need details for next level
        # Ensure even rows for vertical pairing (Lh should have even rows if img had even rows)
        a = Lh[0::2, :]
        b = Lh[1::2, :]
        LL = (a + b) * inv_sqrt2
        return LL

    def feat(self, img: MemoizedImage) -> NDArray[np.uint8]:
        img = self.preproc(img)
        if self.blur:
            img = cv2.GaussianBlur(img, (self.blur, self.blur), 0)
        # iterative DWT levels: repeatedly reduce to LL
        LL = img
        for _ in range(self.levels):
            # ensure even dims before level
            LL = self._ensure_even_dims(LL)
            LL = self._haar_single_level(LL)
        # Zero DC coefficient and compute median excluding DC influence
        flat = LL.flatten()
        flat = flat.copy()
        flat[0] = 0.0  # zero DC (equivalent to excluding it for median)
        return self.bitvec(flat)


class HDiffHash(ImageHash):
    def __init__(self, hash_size: int, edges: bool = False):
        super().__init__((hash_size + 1, hash_size), "gray", edges)

    def feat(self, img: MemoizedImage) -> NDArray[np.uint8]:
        img = self.preproc(img)
        # Hash = binarized horizontal spatial derivative
        diff = img[:, :-1] > img[:, 1:]
        return diff.astype(np.uint8).flatten()


class VDiffHash(ImageHash):
    def __init__(self, hash_size: int, edges: bool = False):
        super().__init__((hash_size, hash_size + 1), "gray", edges)

    def feat(self, img: MemoizedImage) -> NDArray[np.uint8]:
        img = self.preproc(img)
        # Hash = binarized vertical spatial derivative
        diff = img[:-1, :] > img[1:, :]
        return diff.astype(np.uint8).flatten()


class HuMomentHash(ImageHash):
    def __init__(self, img_size, thresh="mean", num_bins: int = 16):
        super().__init__((img_size, img_size), "gray", thresh, True)
        self.num_bins = num_bins

    def feat(self, img: MemoizedImage) -> NDArray[np.uint8]:
        img = self.preproc(img)
        hu = cv2.HuMoments(cv2.moments(img)).flatten()
        log_hu = -np.sign(hu) * np.log10(np.abs(hu), where=np.abs(hu) > 0)
        # "Quantize" the Hu feature vector to a B-bit binary vector
        # Use a tally instead of the binary equivalent of each entry to use hamming distance as a sim metric
        bits = (
            np.sign(log_hu) * np.log2(np.abs(log_hu), where=np.abs(log_hu) > 0) + 1e-8
        ).astype(np.int8)
        bits += self.num_bins // 2
        hash = np.zeros((hu.size, self.num_bins), dtype=np.uint8)
        hash[np.arange(self.num_bins) < bits[:, None]] = 1
        return hash.flatten()
