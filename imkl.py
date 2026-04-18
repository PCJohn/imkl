import yaml
import numpy as np
from numpy.typing import NDArray

from itertools import product
from hashes import *
from utils import MemoizedImage


CV2Img = NDArray[np.uint8]


class MKLClassifier:
    """
    Computes kernel matrices, multiple kernel learning style weights.
    """

    _BINS = np.linspace(-1, 1, 1000, dtype=np.float32)
    _CAT1_LABEL = 1
    _CAT2_LABEL = -1

    def __init__(self, config_file):
        with open(config_file, "r") as f:
            self.cfg = yaml.safe_load(f)
        # Create hash functions with all combinations of shared preproc ops
        preproc_ops = self.cfg["shared_preproc"].keys()
        preproc_val = self.cfg["shared_preproc"].values()
        self.hash_funcs = []
        for kern in self.cfg["kernels"]:
            params = kern["params"]
            for combo in product(*preproc_val):
                params.update(dict(zip(preproc_ops, combo)))
                self.hash_funcs.append(globals()[kern["class"]](**params))
        self.num_hash = len(self.hash_funcs)
        # Sort hash functions by descending order of input image sizes
        self.hash_funcs.sort(key=lambda hf: hf.img_area, reverse=True)
        # Weight vector tracking kernel separability per hash (fit() updates this variable)
        self.weights = np.ones((self.num_hash,), dtype=np.float32) / self.num_hash
        # Params for algorithm used to estimate weights per kernel ("cka" or "rayleigh")
        self.fit_policy = self.cfg["fit_params"].get("policy", "cka")
        self.fit_topk = self.cfg["fit_params"].get("topk", 0)
        self.fit_normalize = self.cfg["fit_params"].get("normalize", True)

    def reset(self):
        self.weights = np.ones((self.num_hash,), dtype=np.float32) / self.num_hash

    def hash(self, imgs: list[CV2Img]):
        """
        Computes hashes for a list of images.

        Args:
            imgs: List of N cv2 images.
        Returns:
            A dict mapping hash function index to an np.uint8 array of shape (N, D).
        """
        imgs = [MemoizedImage(img) for img in imgs]
        return {
            p: np.array([hash.feat(img) for img in imgs], dtype=np.uint8)
            for p, hash in enumerate(self.hash_funcs)
        }

    def combine_kernels(self, kernels: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Computes a weighted sum of a stack of kernels using learned weights.

        Args:
            kernels: An (K, N, N) np.float32 stack of kernel matrices
        Returns:
            An (N, N) matrix that is the weighted sum of the K input matrices.
        """
        return np.einsum("kij,k->ij", kernels, self.weights).astype(np.float32)

    def kernels(
        self,
        hashes: dict[int, NDArray[np.uint8]],
        center: bool = False,
        combine: bool = False,
    ) -> NDArray[np.float32]:
        """
        Computes a stack of kernel matrices using multiple hash functions.

        Args:
            hashes: Dict mapping a hash function id to a list of hash values.
            center: If true, will apply centering to each kernel matrix.
        Returns:
            A (K x N x N) np.float32 array in [-1, 1] that is a stack of similarity matrices.
        """
        if not hashes:
            return np.empty((self.num_hash, 0, 0), dtype=np.float32)
        N = hashes[0].shape[0]
        K = np.array(
            [
                hash_func.hamming_batch(
                    hashes[p], invert=True, gamma=0.0, relative=True
                )
                for p, hash_func in enumerate(self.hash_funcs)
            ],
            dtype=np.float32,
        )
        if center:
            # Apply centering to each kernel
            H = np.eye(N) - np.ones((N, N)) / N
            for p in range(K.shape[0]):
                K[p] = H @ K[p] @ H
        if combine:
            # Return a linear combination of kernels using weights learned in fit()
            return self.combine_kernels(K)
        return K

    def fit(self, imgs_cat1: list[CV2Img], imgs_cat2: list[CV2Img]) -> None:
        """
        Computes weights per kernel indicating how well each "separates" samples from two classes.

        Args:
            imgs_cat1: List of cv2 images of class 1.
            imgs_cat2: List of cv2 images of class 2.
        """
        if self.weights.size == 1:
            # Skip computing weights if we have just one kernel
            return
        imgs = imgs_cat1 + imgs_cat2
        # Make label vector
        Y = np.full(len(imgs), self._CAT1_LABEL)
        Y[len(imgs_cat1) :] = self._CAT2_LABEL
        # Compute centered kernel matrices
        hashes = self.hash(imgs)
        K = self.kernels(hashes, combine=False, center=True)
        # Centered kernel alignment
        L = np.outer(Y, Y)

        # TODO: WIP to mask out label matrix
        # N = L.shape[0]
        # N1 = len(imgs_cat1)
        # mask = np.ones_like(L)
        # mask[N1:, N1:] = 0
        # L *= mask

        # Compute weights indicating how well each kernel "aligns" with the true labels
        if self.fit_policy == "cka":
            cross = np.einsum("kij,ij->k", K, L)
            norm_K = np.einsum("kij,kij->k", K, K)
            norm_L = np.einsum("ij,ij->", L, L)
            self.weights = cross / np.sqrt(norm_K * norm_L)
        elif self.fit_policy == "rayleigh":
            Kf = K.reshape(K.shape[0], -1)  # (K, N*N)
            Lf = L.reshape(-1)  # (N*N,)
            M = Kf @ Kf.T  # (K, K)
            a = Kf @ Lf  # (K,)
            eps = 1e-8
            w = np.linalg.solve(M + eps * np.eye(M.shape[0]), a)
            w = np.maximum(w, 0)
            self.weights = w

        if self.fit_topk:
            thresh = np.partition(self.weights, -self.fit_topk)[-self.fit_topk]
            self.weights[self.weights < thresh] = 0
        if self.fit_normalize:
            self.weights /= self.weights.sum()
