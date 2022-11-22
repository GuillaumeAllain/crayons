import numpy as np
from collections.abc import Callable
from scipy.optimize import root_scalar, root
from dataclasses import dataclass, field, InitVar


@dataclass(frozen=True, eq=True)
class Ray:
    x: InitVar
    y: InitVar
    z: InitVar
    l: InitVar
    m: InitVar
    n: InitVar
    vector: list = field(init=False)

    def __post_init__(self, x, y, z, l, m, n):
        object.__setattr__(self, "vector", np.array([x, y, z, l, m, n]))

    def __getitem__(self, key):
        return self.vector[key]

    @property
    def position(self):
        return self[:3]

    @property
    def cosine(self):
        return self[3:]

    def normalize(self, index: float = 1.0, inplace: bool = False):
        return Ray(*self.position, *self.cosine * index / np.linalg.norm(self.cosine))


def find_intersection(
    ray: Ray,
    sag: Callable[float, float],
    normal: Callable[float, float],
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
    sag: Callable[float, float],
    normal: Callable[float, float],
    t: float = 0,
) -> Ray:
    intersection = find_intersection(ray, sag, normal, t=t)
    if intersection is None:
        return None
    return Ray(*np.concatenate((intersection, ray[3:])))


def refraction(ray: Ray, normal: Callable[float, float], n2: float = 1) -> Ray:
    if np.isclose(np.linalg.norm(ray.vector[3:]), n2):
        return ray
    cosin = ray[3:]
    normalsurf = normal(ray[0], ray[1])
    crossproduct = np.cross(normalsurf, cosin)

    def equation(param):
        param *= n2 / np.linalg.norm(param)
        return crossproduct - np.cross(normalsurf, param)

    solve = root(equation, cosin)
    if solve.success:
        return Ray(*np.concatenate((ray[:3], solve.x * n2 / np.linalg.norm(solve.x))))
    else:
        return None
