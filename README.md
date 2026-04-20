# imkl

CPU-based code to compute multiple perceptual hashes per image. Also computes pairwise Hamming distance matrices.

### Usage

Computing perceptual hashes (each is a binary vector):

```
m = IMKL("config.yaml")

imgs = [img1, img2, ...] # list of cv2 images

# compute a set of binary hashes
hashes = m.hash(imgs)

# compute centered pairwise Hamming distance matrix
K = self.kernels(hashes, combine=False, center=True)

# if you have images from two classes, you can do multiple kernel learning style ops
# and learn weights per hash functions (fit_params in config.yaml has corresponding hyperparameters)
m.fit(train_pos, train_neg)
```

The hash functions and hyperparameters can be modified in the config file (example: `config.yaml`).

### Hash Functions
1. ColorHash (based on [imagehash](https://github.com/JohannesBuchner/imagehash/blob/master/imagehash/__init__.py#L395))
2. PerceptualHash (phash)
3. PixelHash (same as aHash or mHash if you use threshold is "mean" or "median")
4. HDiffHash
5. VDiffHash
6. CornerCountHash
7. GaborHash
8. SqueezeNetHash

### Latency
<img width="2969" height="2063" alt="image" src="https://github.com/user-attachments/assets/f62c9d55-9849-44a5-b0c1-648338fbdf35" />

