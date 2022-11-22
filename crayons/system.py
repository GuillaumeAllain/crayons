from dataclasses import dataclass, field
from collections.abc import Iterable
from .materials import Material
from .propagation import transfert, refraction, Ray
from .surfaces import surfaces_catalog
import numpy as np
import tabulate


@dataclass(frozen=False)
class Surface:
    type: str = "sph"
    comment: str = ""
    args: dict = field(default_factory=dict)
    thickness: float = 0
    material: Material = field(default_factory=lambda: Material(name="air"))
    sag_func: dict = field(default=None)

    def __post_init__(self):
        assert self.type in surfaces_catalog.keys(), "Surface type must be in catalo"
        assert self.material, "Material must be specified"
        object.__setattr__(self, "sag_func", surfaces_catalog[self.type])
        # TODO add support for custom surfaces
        # TODO assert kwargs


@dataclass(repr=True)
class System:
    wavelengths: float or Iterable = 587.5618
    stop: int = 1
    surfaces: list[Surface] = field(
        default_factory=lambda: list(
            [
                Surface(comment="Object", args={"c": 0}),
                Surface(comment="Stop", args={"c": 0}),
                Surface(comment="Image", args={"c": 0}),
            ]
        )
    )

    def __repr__(self):
        data_print = "System data\n" + tabulate.tabulate(
            [
                [
                    "Wavelengths (nm)",
                    self.wavelengths,
                ]
            ],
            headers=["Data", "Values"],
            tablefmt="fancy_grid",
        )
        surface_print = "\nSurface data\n" + tabulate.tabulate(
            [
                [
                    "stop" if i == self.stop else i,
                    s.type,
                    s.thickness,
                    s.material,
                    s.args,
                    s.comment,
                ]
                for i, s in enumerate(self.surfaces)
            ],
            headers=[" ", "Type", "Thick.", "Mat.", "Args", "Com."],
            tablefmt="fancy_grid",
        )
        return data_print + surface_print

    def __getitem__(self, key):
        return self.surfaces[key]

    def __setitem__(self, key, value):
        object.__setattr__(
            self, "surfaces", self.surfaces[:key] + [value] + self.surfaces[key + 1 :]
        )

    def __len__(self):
        return len(self.surfaces)

    def reverse(self):
        object.__setattr__(
            self,
            "surfaces",
            list(reversed(self.surfaces)),
        )

    def insert(self, key, value):
        object.__setattr__(
            self, "surfaces", self.surfaces[:key] + [value] + self.surfaces[key:]
        )

    def pop(self, key):
        object.__setattr__(
            self, "surfaces", self.surfaces[:key] + self.surfaces[key + 1 :]
        )

    def propagate(self, ray: tuple, key: int = None, reverse=False):
        if key is None:
            key = 0
        propagation_array = []
        range_obj = (
            range(key - 1, -1, -1) if reverse else range(key + 1, len(self.surfaces))
        )

        ray = Ray(
            *ray.vector[:2],
            self.surfaces[key].sag_func["sag"](
                *ray.vector[:2], **self.surfaces[key].args
            ),
            *ray.vector[3:],
        )
        ray.normalize(self.surfaces[key].material.index(self.wavelengths))
        propagation_array.append(ray.vector)
        for curr in range_obj:
            current_surface_func = self.surfaces[curr].sag_func
            ray = transfert(
                ray,
                lambda x, y: current_surface_func["sag"](
                    x, y, **self.surfaces[curr if reverse else curr].args
                ),
                lambda x, y: current_surface_func["normal"](
                    x, y, **self.surfaces[curr if reverse else curr].args
                ),
                (-1 if reverse else 1)
                * self.surfaces[curr if reverse else curr - 1].thickness,
            )
            if reverse:
                propagation_array.append(ray.vector)
            ray = refraction(
                ray,
                lambda x, y: current_surface_func["normal"](
                    x,
                    y,
                    **self.surfaces[curr if reverse else curr].args,
                ),
                n2=self.surfaces[
                    (curr - 1 if curr > 0 else 0) if reverse else curr
                ].material.index(self.wavelengths),
            )
            if not reverse:
                propagation_array.append(ray.vector)
        return np.array(propagation_array)[:: -1 if reverse else 1]
