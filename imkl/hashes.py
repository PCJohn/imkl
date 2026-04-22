from abc import ABC, abstractmethod
import cv2
import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort

from .utils import ImagePreprocTransforms, MemoizedImage


class ImageHash(ABC):
    _THRESH_FUNCS = {
        "mean": np.mean,
        "median": np.median,
        "p50": np.median,
        "p90": lambda x: np.percentile(x, 90),
        "p95": lambda x: np.percentile(x, 95),
        "p99": lambda x: np.percentile(x, 99),
    }
    _MAX_HASH_DIM = 512
    _COUNT_LUT = np.array(
        [bin(i).count("1") for i in range(_MAX_HASH_DIM)], dtype=np.uint8
    )

    def __init__(
        self,
        img_size: tuple[int, int],
        hash_dim: int,
        col: str,
        thresh: str = "mean",
        edges: bool = False,
        log_polar: bool = False,
    ):
        self.preproc_transform = ImagePreprocTransforms(img_size, col, edges, log_polar)
        self.img_area = img_size[0] * img_size[1]
        self.hash_dim = hash_dim
        self.thresh = thresh
        self.thresh_func = self._THRESH_FUNCS.get(thresh)

    def preproc(self, img: MemoizedImage):
        # Resize, set color space and cast to fp32
        return img.preproc(self.preproc_transform).astype(np.float32)

    def bitvec(self, x: NDArray[np.float32], thresh: bool = True) -> NDArray[np.uint8]:
        if not thresh:
            return np.packbits(x)
        # Threshold features and binarize to bit vectors
        bits = x > self.thresh_func(x)
        return np.packbits(bits)

    def count_to_bitvec(self, count: int) -> NDArray[np.uint8]:
        # Convert integer counts into bit vectors = tally of log2(count)
        feat = np.zeros((self.hash_dim,), dtype=np.uint8)
        feat[: int(np.log2(count))] = 1
        return np.packbits(feat)

    @abstractmethod
    def feat(self, img: MemoizedImage) -> NDArray[np.uint8]:
        # Must extract a perceptual hash (a binary feature vector) of an image
        pass

    def hamming_batch(
        self,
        x: NDArray[np.uint8],
        invert: bool = False,
        gamma: float = 0.0,
        relative: bool = False,
    ) -> NDArray[np.float32]:
        """
        Compute a matrix with pairwise Hamming distances given binary vectors.

        Args:
            x: A (N, D) binary numpy array set of binary vectors.
            gamma: Coefficient for exponentiated hamming distance. Default: 0 (no exponentiation).
            relative: If true, returns relative Hamming distance = hamming / D. Default: false.
        Returns:
            A float32 (N, N) numpy array with pairwise Hamming distance between input vectors.
        """
        xor = x[:, np.newaxis, :] ^ x[np.newaxis, :, :]
        out = self._COUNT_LUT[xor].sum(axis=2).astype(np.float32)
        if invert:
            out = self.hash_dim - out
        if relative:
            out /= self.hash_dim
        if gamma:
            out = np.exp(-gamma * out)
        return out


class ColorHash(ImageHash):
    def __init__(
        self,
        img_size: int,
        edges: bool = False,
        log_polar: bool = False,
        binbits: int = 3,
    ):
        super().__init__(
            (img_size, img_size), 14 * binbits, "bgr", "mean", edges, log_polar
        )
        self.max_val = 2**binbits
        self.shifts = np.arange(binbits - 1, -1, -1, dtype=np.uint8)

    def feat(self, img: MemoizedImage):
        """
        Color hash based on colorhash in https://github.com/JohannesBuchner/imagehash
        Ref: https://github.com/JohannesBuchner/imagehash/blob/master/imagehash/__init__.py#L395

        Args:
            img (MemoizedImage): Image to hash
        Returns:
            A binary numpy array.
        """
        img = self.preproc(img).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_raw, s, v = cv2.split(hsv)
        # Opencv has the Hue channel in range [0, 180]
        h = (h_raw.astype(np.float32) / 179 * 255).astype(np.uint8).flatten()
        s = s.flatten()
        mask_black = gray < 32  # 32 = 256 // 8
        frac_black = np.mean(mask_black)
        mask_gray = s < 85  # 85 = 256 // 3
        frac_gray = np.logical_and(~mask_black, mask_gray).mean()
        mask_colors = np.logical_and(~mask_black, ~mask_gray)
        mask_faint = np.logical_and(mask_colors, s < 170)  # 170 = 256 * 2 // 3
        mask_bright = np.logical_and(mask_colors, s > 170)  # 170 = 256 * 2 // 3
        hue_bins = np.linspace(0, 255, 7)  # 7 = 6 + 1 for 6 bins
        num_hue_bins = len(hue_bins) - 1
        c = max(1, mask_colors.sum())
        h_faint_counts = (
            np.zeros(num_hue_bins)
            if not mask_faint.any()
            else np.histogram(h[mask_faint], bins=hue_bins)[0]
        )
        h_bright_counts = (
            np.zeros(num_hue_bins)
            if not mask_bright.any()
            else np.histogram(h[mask_bright], bins=hue_bins)[0]
        )
        raw_values = np.concatenate(
            ([frac_black, frac_gray], h_faint_counts / c, h_bright_counts / c)
        )
        values = np.clip(
            np.floor(raw_values * self.max_val), 0, self.max_val - 1
        ).astype(np.uint8)
        bits = (values[:, np.newaxis] >> self.shifts) & 1
        return self.bitvec(bits, thresh=False)


class GaborHash(ImageHash):
    def __init__(
        self,
        img_size: int,
        hash_dim: int,
        model_path: str,
        thresh: str,
        edges: bool = False,
        log_polar: bool = False,
    ):
        super().__init__(
            (img_size, img_size), hash_dim, "gray", thresh, edges, log_polar
        )
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
    def __init__(
        self,
        img_size: int,
        hash_dim: int,
        model_path: str,
        thresh: str,
        edges: bool = False,
        log_polar: bool = False,
    ):
        super().__init__(
            (img_size, img_size), hash_dim, "col", thresh, edges, log_polar
        )
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def feat(self, img: MemoizedImage) -> NDArray[np.uint8]:
        img = self.preproc(img)
        img = img.transpose(2, 0, 1)[None, :, :, :]
        f = self.sess.run(None, {self.input_name: img})[0][0]
        return self.bitvec(f)


class PerceptualHash(ImageHash):
    def __init__(
        self,
        hash_size: int,
        highfreq_factor: int,
        thresh: str,
        edges: bool = False,
        log_polar: bool = False,
    ):
        super().__init__(
            (hash_size * highfreq_factor, hash_size * highfreq_factor),
            hash_size**2,
            "gray",
            thresh,
            edges,
            log_polar,
        )
        self.hash_size = hash_size

    def feat(self, img: MemoizedImage) -> NDArray[np.uint8]:
        img = self.preproc(img)
        dct_low_freq = cv2.dct(img)[: self.hash_size, : self.hash_size]
        return self.bitvec(dct_low_freq)


class PixelHash(ImageHash):
    def __init__(
        self, hash_size: int, thresh: str, edges: bool = False, log_polar: bool = False
    ):
        super().__init__(
            (hash_size, hash_size), hash_size**2, "gray", thresh, edges, log_polar
        )

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
        log_polar: bool = False,
    ):
        super().__init__(
            (hash_size * scale, hash_size * scale),
            None,
            "gray",
            thresh,
            edges,
            log_polar,
        )
        self.levels = int(np.log2(scale))
        # FIXME: inferring hash_dim this way might not work for img with odd sized dims
        self.hash_dim = (hash_size * scale // 2**self.levels) ** 2
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
    def __init__(self, hash_size: int, edges: bool = False, log_polar: bool = False):
        super().__init__(
            (hash_size, hash_size + 1), hash_size**2, "gray", edges, log_polar
        )

    def feat(self, img: MemoizedImage) -> NDArray[np.uint8]:
        img = self.preproc(img)
        return self.bitvec(
            (img[:, 1:] > img[:, :-1]).astype(np.uint8).flatten(), thresh=False
        )


class VDiffHash(ImageHash):
    def __init__(self, hash_size: int, edges: bool = False, log_polar: bool = False):
        super().__init__(
            (hash_size + 1, hash_size), hash_size**2, "gray", edges, log_polar
        )

    def feat(self, img: MemoizedImage) -> NDArray[np.uint8]:
        img = self.preproc(img)
        return self.bitvec(
            (img[1:, :] > img[:-1, :]).astype(np.uint8).flatten(), thresh=False
        )


class HOGHash(ImageHash):
    def __init__(
        self,
        img_size: int,
        thresh: str,
        edges: bool = False,
        log_polar: bool = False,
        num_bins: int = 4,
    ):
        super().__init__((img_size, img_size), None, "gray", thresh, edges, log_polar)
        win = (64, 64)
        block = (32, 32)
        stride = (32, 32)
        cell = (16, 16)
        self.hog = cv2.HOGDescriptor(
            _winSize=win,
            _blockSize=block,  # num blocks = (_winSize / _blockSize) ** 2
            _blockStride=stride,  # tiled blocks
            _cellSize=cell,  # num cells = (_blockSize / _cellSize) ** 2
            _nbins=num_bins,
        )
        # Hash dim = Descriptor size = num blocks * num cells * num bins
        self.hash_dim = (
            ((win[0] - block[0]) // stride[0] + 1)
            * ((win[1] - block[1]) // stride[1] + 1)
            * (block[0] // cell[0])
            * (block[1] // cell[1])
            * num_bins
        )

    def feat(self, img: MemoizedImage) -> NDArray[np.uint8]:
        img = self.preproc(img).astype(np.uint8)
        return self.bitvec(self.hog.compute(img))


class CornerCountHash(ImageHash):
    def __init__(
        self, img_size: int, hash_dim: int, edges: bool = False, log_polar: bool = False
    ):
        super().__init__((img_size, img_size), hash_dim, "gray", edges, log_polar)
        self.fast_feat_det = cv2.FastFeatureDetector_create(
            threshold=10, nonmaxSuppression=True
        )

    def feat(self, img: MemoizedImage) -> NDArray[np.uint8]:
        img = self.preproc(img).astype(np.uint8)
        corners = self.fast_feat_det.detect(img)
        return self.count_to_bitvec(len(corners) + 1)


class LineCountHash(ImageHash):
    def __init__(
        self, img_size: int, hash_dim: int, edges: bool = False, log_polar: bool = False
    ):
        super().__init__((img_size, img_size), hash_dim, "gray", edges, log_polar)
        self.lsd = cv2.createLineSegmentDetector()

    def feat(self, img: MemoizedImage) -> NDArray[np.uint8]:
        img = self.preproc(img).astype(np.uint8)
        lines = self.lsd.detect(img)[0]
        num_lines = 1 if lines is None else lines.shape[0]
        return self.count_to_bitvec(num_lines + 1)
