# imkl
Code to compute kernel matrices and train small classifiers fast on CPUs

### Usage

```
from imkl import MKLClassifier

model = MKLClassifier("config.yaml")
model.fit(pos_imgs, neg_imgs)
pred = model.predict(test_imgs)
```