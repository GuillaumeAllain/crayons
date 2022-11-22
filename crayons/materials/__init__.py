from dataclasses import dataclass, field
from collections.abc import Iterable
from functools import cache
import numpy as np
from os import linesep


@dataclass(frozen=True, eq=True)
class Material:
    name: str = field(default=None)
    code: str = field(default=None)
    B: tuple = field(default=None)
    C: tuple = field(default=None)
    n: float = field(default=None)
    vd: float = field(default=None)

    def __post_init__(self):
        assert (
            (self.code or self.name) or (self.B and self.C) or (self.n)
        ), "Either name or code must be specified"
        if not self.name and self.code:
            object.__setattr__(self, "name", self.code)
        if self.name is not None and self.code is None:
            if self.name == "air":
                object.__setattr__(self, "n", 1.0)
            else:
                raise NotImplementedError("Material code lookup is not implemented")
        if ((not self.n) and (not self.vd)) and not (self.B and self.C):
            n, vd = self.code.split(":")
            n, vd = float("1." + n), float(vd[:2] + "." + vd[2:])
            object.__setattr__(self, "n", n)
            object.__setattr__(self, "vd", vd)

    def __repr__(self):
        return (
            self.name
            if self.name
            else self.code
            if self.code
            else f"n: {self.n}{(linesep+'vd: '+str(self.vd) if self.vd else '')}"
            if self.n
            else "Sellmeier"
        )

    def index(self, lam):
        return refractive_index(lam, self)


@cache
def sellmeier(lam: float, B: Iterable, C: Iterable):
    assert np.shape(B) == np.shape(C)
    return np.sqrt(1 + sum(b * lam**2 / (lam**2 - c) for b, c in zip(B, C)))


@cache
def index_abbe(lam: float, nd: float, vd: float) -> float:
    return nd + ((lam - 589.3) * (1 - nd) / (170.2 * vd)) / 1000


@cache
def refractive_index(lam: float, material: Material) -> float:
    if material.B and material.C:
        return sellmeier(lam, material.B, material.C)
    elif material.n and not material.vd:
        return material.n
    elif material.n and material.vd:
        return index_abbe(lam, material.n, material.vd)
