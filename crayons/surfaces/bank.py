import numpy as np


def __sag_sph(x, y, c=0):
    assert np.shape(x) == np.shape(y)
    if c == 0:
        return np.zeros_like(x)
    cradiussq = c * (np.asarray(x) ** 2 + np.asarray(y) ** 2)
    root = 1 - c * cradiussq
    with np.errstate(invalid="ignore"):
        return np.where(root > 0, cradiussq / (1 + np.sqrt(root)), np.nan)


def __sag_sph_norm(x, y, c=0):
    assert np.shape(x) == np.shape(y)
    if c == 0:
        return np.array([
            np.zeros_like(x),
            np.zeros_like(x),
            np.ones_like(x) * -1,
        ])
    rad = 1 - c**2 * (np.asarray(x) ** 2 + np.asarray(y) ** 2)
    srad = np.where(rad > 0, np.sqrt(rad), np.nan)
    com = c * (2 * srad + 1 + rad) / (srad * (srad + 1) ** 2)  # common between x and y
    vec = np.array([x * com, y * com, np.ones_like(x) * -1])
    vec /= np.linalg.norm(vec, axis=0)  # normalisation
    return vec


surfaces_catalog = {"sph": {"kwargs": ["c"], "sag": __sag_sph, "normal": __sag_sph_norm}}
