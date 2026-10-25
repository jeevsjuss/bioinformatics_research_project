"""Save and load fitted scikit-learn pipelines as pickle files.

Each model is stored as a small dictionary ("bundle") containing:
  - the fitted sklearn Pipeline (binarise -> variance filter -> feature selection -> classifier)
  - the LabelEncoder, where labels were integer-encoded before fitting (PT models)
  - the package versions used to train it
  - a SHA-256 checksum is written next to the file (<name>.pkl.sha256)

The fitted pipeline remembers the exact k-mer column names and order it was
trained on (pipeline.feature_names_in_), so `predict()` below aligns any new
k-mer matrix to those columns automatically.

SECURITY: pickle files can run arbitrary code when loaded. Only load model
files from sources you trust, and check the SHA-256 checksum.

COMPATIBILITY: pickles are tied to the scikit-learn version used to create them.
Load them with the versions recorded in the bundle (see requirements.txt).
"""
import hashlib
import os
import pickle
import platform
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn


def _versions():
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model(pipeline, path, label_encoder=None, description=""):
    """Pickle a fitted pipeline (and its LabelEncoder, if any) to `path`.

    Returns the SHA-256 checksum, which is also written to `path + '.sha256'`.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    classes = label_encoder.classes_ if label_encoder is not None else pipeline.classes_
    bundle = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "classes": np.asarray(classes).tolist(),
        "n_features_in": int(pipeline.n_features_in_),
        "versions": _versions(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": description,
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=4)
    digest = _sha256(path)
    with open(path + ".sha256", "w") as f:
        f.write(f"{digest}  {os.path.basename(path)}\n")
    print(f"Saved model to {path} (sha256 {digest[:12]}...)")
    return digest


def load_model(path, verify_checksum=True):
    """Load a bundle saved by save_model(). Only load files you trust."""
    checksum_file = path + ".sha256"
    if verify_checksum and os.path.exists(checksum_file):
        expected = open(checksum_file).read().split()[0]
        if _sha256(path) != expected:
            raise ValueError(f"Checksum mismatch for {path}; the file may be corrupted or altered.")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    trained = bundle["versions"]["scikit-learn"]
    if trained != sklearn.__version__:
        warnings.warn(
            f"Model was trained with scikit-learn {trained} but you have "
            f"{sklearn.__version__}; results may differ or loading may fail."
        )
    return bundle


def predict(bundle, X):
    """Predict labels for a DataFrame X (rows = isolates, columns = k-mers).

    Columns are aligned to the training k-mers: k-mers absent from X are
    treated as count 0 and extra k-mers are ignored.
    """
    pipe = bundle["pipeline"]
    expected = list(pipe.feature_names_in_)
    n_missing = len(set(expected) - set(X.columns))
    if n_missing:
        warnings.warn(f"{n_missing} of {len(expected)} training k-mers are missing from X; filled with 0.")
    X = X.reindex(columns=expected, fill_value=0)
    raw = pipe.predict(X)
    le = bundle["label_encoder"]
    return le.inverse_transform(raw) if le is not None else raw
