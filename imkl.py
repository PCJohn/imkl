import yaml
import numpy as np
from numpy.typing import NDArray

from itertools import product
from hashes import *
from utils import multi_hist, multi_otsu, softmax, MemoizedImage


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
        # Create hash functions with all combinations of shared preprocs
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
        # Weight vector tracking kernel separability per hash function
        self.weights = np.ones((self.num_hash,), dtype=np.float32) / self.num_hash

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
        return np.einsum("ijk,i->jk", kernels, self.weights).astype(np.float32)

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
        K = np.zeros([self.num_hash, N, N], dtype=np.float32)
        [
            hash_func.sim_batch(hashes[p], out=K[p])
            for p, hash_func in enumerate(self.hash_funcs)
        ]
        if center:
            # Apply cetering to each kernel
            I = np.eye(N)
            I_N = I - np.ones((N, N)) / N
            for p in range(K.shape[0]):
                K[p] = I_N @ K[p] @ I_N
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
        # Add augmentations
        imgs = imgs_cat1 + imgs_cat2
        # Make label vector
        Y = np.full(len(imgs), self._CAT1_LABEL)
        Y[len(imgs_cat1) :] = self._CAT2_LABEL
        # Compute centered kernel matrices
        hashes = self.hash(imgs)
        K = self.kernels(hashes, combine=False, center=True)
        # Centered kernel alignment
        label = np.outer(Y, Y)
        cross = np.einsum("knn,nn->k", K, label)
        norm_K = np.sqrt(np.einsum("knn,knn->k", K, K))
        norm_L = np.linalg.norm(label, ord="fro")
        self.weights = cross / (norm_K * norm_L)

    def fit_models(self, hashes, labels):
        """
        Fit classification models.

        Args:
            hashes: A (N, D) binary numpy array with perceptual hashes.
            labels: A numpy array with target binary classification labels.
        """
        from sklearn.naive_bayes import BernoulliNB

        self.models = [BernoulliNB() for _ in range(self.num_hash)]
        # Train classifiers with different types of hashes
        for p in range(self.num_hash):
            self.models[p].fit(np.array(hashes[p]), labels)

    def predict(self, imgs: list[CV2Img]) -> NDArray[np.float32]:
        """
        Predict classification labels.
        (1) Predicts labels with classifiers trained on top of each hash functions.
        (2) Uses weights learned in fit() as weights to compute linear combination of predictions.

        Args:
            imgs: List of N images to classify.
        Returns:
            An (N,) shaped numpy array with predicted class labels.
        """
        if not imgs:
            return np.empty((0,), dtype=np.float32)
        hashes = self.hash(imgs)
        pY = [
            self.models[p].predict_proba(hashes[p])[:, 1] for p in range(self.num_hash)
        ]
        return np.einsum("ij,i->j", pY, self.weights).astype(np.float32)
