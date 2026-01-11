import sys, time, pickle, yaml

import cv2
import numpy as np
from numpy.typing import NDArray

from hashes import *
from utils import multi_hist, multi_otsu, softmax, MemoizedImage

from sklearn.naive_bayes import BernoulliNB


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
        self.hash_funcs = [
            globals()[c["class"]](**c["params"]) for c in self.cfg["kernels"]
        ]
        self.num_hash = len(self.hash_funcs)
        # Sort hash functions by descending order of input image sizes
        self.hash_funcs.sort(key=lambda hf: hf.img_area, reverse=True)
        self.max_img_size = self.hash_funcs[0].img_size
        # Weight vector tracking kernel separability per hash function
        self.weights = np.ones((self.num_hash,), dtype=np.float32) / self.num_hash
        # Actual classification models
        self.models = [BernoulliNB() for _ in range(self.num_hash)]

    def reset(self):
        # Resets kernel weights.
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

    def compute_all_kernels(
        self, hashes: dict[int, NDArray[np.uint8]], center: bool = False
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
        K = np.zeros([self.num_hash, N, N], dtype=np.float32)
        [
            hash_func.sim_batch(hashes[p], out=K[p])
            for p, hash_func in enumerate(self.hash_funcs)
        ]
        # Rescale to [-1, 1] as hash sim is in [0, 1]
        K *= 2
        K -= 1
        if center:
            I = np.eye(N)
            I_N = I - np.ones((N, N)) / N
            for p in range(K.shape[0]):
                K[p] = I_N @ K[p] @ I_N
        return K

    def fit(
        self, imgs_cat1: list[CV2Img], imgs_cat2: list[CV2Img], method: str = "otsu"
    ) -> None:
        """
        Computes weights per kernel indicating how well each "separates" samples from two classes.

        Args:
            imgs_cat1: List of cv2 images of class 1.
            imgs_cat2: List of cv2 images of class 2.
            method: Which measure to use to estimate separability per kernel. Must be one of ["otsu", "align", "mmd"] (default: "otsu").
            normalize: If true, normalizes weights to sum to 1. Default = True.
        """
        # Add augmentations
        imgs = imgs_cat1 + imgs_cat2
        # Make label vector
        Y = np.full(len(imgs), self._CAT1_LABEL)
        Y[len(imgs_cat1) :] = self._CAT2_LABEL
        # Compute kernel matrices
        hashes = self.hash(imgs)
        K = self.compute_all_kernels(hashes)
        # Compute kernel separability
        if method == "alignment":
            # Weight kernels based on alignment with label kernel
            pass
        elif method == "mmd":
            # Weight kernels based on max mean discrepancy
            pass
        elif method == "otsu":
            # Kernel separability using otsu threshold (hacky, but smells like a special case of the general kernel based sample test)
            # intra class similarities
            intra_sim_mask = np.ix_(Y == self._CAT1_LABEL, Y == self._CAT1_LABEL)
            # inter class similarities
            inter_sim_mask = np.ix_(Y == self._CAT1_LABEL, Y == self._CAT2_LABEL)
            # Histograms of intra and inter class distances
            intra_sim_hist = multi_hist(K, intra_sim_mask, self._BINS)
            inter_sim_hist = multi_hist(K, inter_sim_mask, self._BINS)
            # Otsu threshold ideal for the combined histogram (regardless of labels)
            # Use the Otsu effectiveness score as weight
            combined_hist = intra_sim_hist + inter_sim_hist
            thresh, effectiveness = multi_otsu(combined_hist, self._BINS)
            self.weights = effectiveness
        # Normalize weights
        self.weights = softmax(self.weights)
        # self.weights /= self.weights.sum()
        # Train classifiers with different types of hashes
        for p in range(self.num_hash):
            self.models[p].fit(np.array(hashes[p]), Y)

    def predict(self, imgs: list[CV2Img]) -> NDArray[np.float32]:
        """
        Predict classification labels.
        (1) Predicts labels with classifiers trained on top of each hash functions.
        (2) Uses weights learned in fit() as weights to compute linear combination of predictions.

        Args:
            imgs: List of N images to classify.
        Return:
            An (N,) shaped numpy array with predicted class labels.
        """
        if not imgs:
            return np.empty((0,), dtype=np.float32)
        hashes = self.hash(imgs)
        pY = [
            self.models[p].predict_proba(hashes[p])[:, 1] for p in range(self.num_hash)
        ]
        return np.einsum("ij,i->j", pY, self.weights).astype(np.float32)
