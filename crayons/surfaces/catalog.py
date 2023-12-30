import numpy as np
from scipy.spatial.transform import Rotation as R


# def add_tilt_sag(func):
#     def wrapper(x, y, **kwargs):
#         if np.all(kwargs["rotation"] == [0, 0, 0]):
#             points = np.c_[x, y, np.zeros_like(x)]
#         else:
#             tilt = kwargs["rotation"]
#             kwargs.pop("rotation", None)
#             points = R.from_euler("xyz", tilt, degrees=True).apply(
#                 np.c_[x, y, np.zeros_like(x)]
#             )
#         return func(*points.T[:2], **kwargs) + points.T[2]
#
#     return wrapper


def rectangular_aperture(x, y, rex, rey):
    return np.clip(x, -rex, rex), np.clip(y, -rey, rey)


def circular_aperture(x, y, cir):
    return np.where(x**2 + y**2 <= cir**2, x, np.sign(x) * cir), np.where(
        x**2 + y**2 <= cir**2, y, np.sign(y) * cir
    )


def add_aperture(func):
    def wrapper(x, y, **kwargs):
        if "aperture" in kwargs.keys():
            x_edge, y_edge = x, y
            for aperture in kwargs["aperture"]:
                if aperture["type"] == "rectangular":
                    x_edge, y_edge = rectangular_aperture(
                        x_edge, y_edge, aperture["rex"], aperture["rey"]
                    )
                if aperture["type"] == "circular":
                    x_edge, y_edge = circular_aperture(x_edge, y_edge, aperture["cir"])
            return func(x_edge, y_edge, **kwargs)

        else:
            return func(x, y, **kwargs)

    return wrapper


@add_aperture
def add_tilt_norm(func):
    def wrapper(x, y, **kwargs):
        if np.all(kwargs["rotation"] == [0, 0, 0]):
            points = np.c_[x, y, np.zeros_like(x)]
        else:
            tilt = kwargs["rotation"]
            kwargs.pop("rotation", None)
            points = R.from_euler("xyz", tilt, degrees=True).apply(
                np.c_[x, y, np.zeros_like(x)]
            )
        return func(*points.T[:2], **kwargs).T

    return wrapper


@add_aperture
def __sag_sph(x, y, c=0, **kwargs):
    assert np.shape(x) == np.shape(y)
    if c == 0:
        return np.zeros_like(x)
    cradiussq = c * (np.asarray(x) ** 2 + np.asarray(y) ** 2)
    root = 1 - c * cradiussq
    with np.errstate(invalid="ignore"):
        return np.where(root > 0, cradiussq / (1 + np.sqrt(root)), np.nan)


@add_aperture
def __sag_asp(x, y, c=0, k=0, coef=None, **kwargs):
    assert np.shape(x) == np.shape(y)
    if c == 0:
        return np.zeros_like(x)
    if coef is None and k == 0:
        return __sag_sph(x, y, c=c, **kwargs)
    radius2 = np.asarray(x) ** 2 + np.asarray(y) ** 2
    root = 1 - ((1 + k) * c**2 * radius2)
    with np.errstate(invalid="ignore"):
        return np.where(
            root > 0,
            ((c * radius2) / (1 + np.sqrt(root)))
            + (
                np.dot(
                    coef,
                    np.power(
                        np.tile(radius2, [len(coef), 1])
                        * np.ones_like(coef).reshape(-1, 1),
                        (np.arange(len(coef)) + 1).reshape(-1, 1),
                    ),
                )
                if coef is not None
                else 0
            ),
            np.nan,
        )


@add_aperture
def __sag_sph_norm(x, y, c=0, **kwargs):
    assert np.shape(x) == np.shape(y)
    if c == 0:
        return np.array(
            [
                np.zeros_like(x),
                np.zeros_like(x),
                np.ones_like(x) * -1,
            ]
        )
    rad = 1 - c**2 * (np.asarray(x) ** 2 + np.asarray(y) ** 2)
    srad = np.where(rad > 0, np.sqrt(rad), np.nan)
    com = c * (2 * srad + 1 + rad) / (srad * (srad + 1) ** 2)  # common between x and y
    vec = np.array([x * com, y * com, np.ones_like(x) * -1])
    vec /= np.linalg.norm(vec, axis=0)  # normalisation
    return vec


surfaces_catalog = {
    "sph": {"kwargs": ["c"], "sag": __sag_sph, "normal": __sag_sph_norm},
    "asp": {"kwargs": ["c"], "sag": __sag_asp, "normal": __sag_sph_norm},
}
