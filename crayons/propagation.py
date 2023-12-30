import numpy as np
from collections.abc import Callable, Iterable
from scipy.optimize import root_scalar, root
from dataclasses import dataclass, field, InitVar
from warnings import warn


@dataclass(frozen=True, eq=True)
class Ray:
    x: InitVar
    y: InitVar
    z: InitVar
    l: InitVar
    m: InitVar
    n: InitVar
    wavelength: float = field(default=None)
    vector: list = field(init=False)

    def __post_init__(self, x, y, z, l, m, n):
        object.__setattr__(self, "vector", np.array([x, y, z, l, m, n]))

    def __getitem__(self, key):
        return self.vector[key]

    def __setitem__(self, key, value):
        self.vector[key] = value

    @property
    def position(self):
        return self[:3]

    @property
    def cosine(self):
        return self[3:]

    def normalize(self, index: float = 1.0, inplace: bool = True):
        vec = np.array(
            [*self.position, *self.cosine * index / np.linalg.norm(self.cosine)]
        )
        if inplace:
            object.__setattr__(self, "vector", vec)
        else:
            return Ray(*vec)

    def __mul__(self, other):
        return Ray(*self.vector * other)


def find_intersection(
    ray: Ray,
    sag: Callable,
    normal: Callable,
    t: float = 0,
) -> np.ndarray:
    position = np.asarray(ray[:3])
    angle = np.asarray(ray[3:])

    def equation(param: int):
        return (
            position[2]
            - t
            + param * angle[2]
            - sag(position[0] + param * angle[0], position[1] + param * angle[1])
        ), angle[2] - np.sum(
            normal(position[0] + param * angle[0], position[1] + param * angle[1])[:2]
            * angle[:2]
        )

    solve = root_scalar(equation, x0=t, fprime=True)
    if solve.converged:
        return (solve.root) * angle + position - np.array((0, 0, t))
    else:
        return None


def transfert(
    ray: Ray,
    sag: Callable,
    normal: Callable,
    t: float = 0,
) -> Ray:
    intersection = find_intersection(ray, sag, normal, t=t)
    if intersection is None:
        return None
    return Ray(*np.concatenate((intersection, ray[3:])), ray.wavelength)


def refraction(ray: Ray, normal: Iterable[3], n2: float = 1) -> Ray:
    if np.isclose(np.linalg.norm(ray.vector[3:]), n2):
        return ray
    mu = np.linalg.norm(ray[3:]) / n2
    cosin = ray[3:] / (mu * n2)
    product_normal = np.dot(normal, cosin)
    try:
        cosout = np.sqrt(1 - mu**2 * (1 - product_normal**2)) * normal + mu * (
            product_normal * normal - cosin
        )
    except ValueError:
        warn("Total reflection")
        return None
    return Ray(*np.r_[ray[:3], -cosout * n2], ray.wavelength)
