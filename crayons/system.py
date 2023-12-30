from dataclasses import dataclass, field
from collections.abc import Iterable
from .materials import Material
from .propagation import transfert, refraction, Ray
from copy import copy
from .surfaces import surfaces_catalog
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import tabulate

# @dataclass(frozen=True)
# class Aperture:
#     pass


@dataclass(frozen=False)
class Surface:
    type: str = "sph"
    comment: str = ""
    args: dict = field(default_factory=dict)
    thickness: float = 0
    material: Material = field(default_factory=lambda: Material(name="air"))
    sag_func: dict = field(default=None)
    positionning: str = "loc"

    def __post_init__(self):
        assert self.type in surfaces_catalog.keys(), "Surface type must be in catalog"
        assert self.material, "Material must be specified"
        object.__setattr__(self, "sag_func", surfaces_catalog[self.type])
        # TODO add support for custom surfaces
        # TODO assert kwargs
        default_args = {
            "decenter": np.array([0, 0, 0]),
            "rotation": np.array([0, 0, 0]),
            "aperture": list(),
        }
        default_args.update(self.args)
        self.args = default_args

    @property
    def direction_cosine(self):
        return R.from_euler("xyz", self.args["rotation"], degrees=True).apply([0, 0, 1])


@dataclass(repr=True)
class System:
    stop: int = 1
    wavelengths: float or Iterable = field(
        default_factory=lambda: [
            587.5618,
        ]
    )
    surface_pointer = 0
    reference_wavelength: int = 0
    wavelengths_weights: float or Iterable = field(
        default_factory=lambda: [
            1,
        ]
    )
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
                    str(self.wavelengths),
                ],
                ["Wavelengths  weight", str(self.wavelengths_weights)],
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
                    "\n".join(
                        [f"{x[0]}:{x[1]}" for x in zip(s.args.keys(), s.args.values())]
                    ),
                    s.positionning,
                    s.comment,
                ]
                for i, s in enumerate(self.surfaces)
            ],
            headers=[" ", "Type", "Thick.", "Mat.", "Args", "Pos", "Com."],
            tablefmt="fancy_grid",
        )
        return data_print + surface_print

    @property
    def wavelengths(self):
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        self._wavelengths = np.atleast_1d(value)
        self._wavelengths_weights = np.ones_like(self.wavelengths)

    @property
    def wavelengths_weights(self):
        return self._wavelengths_weights

    @wavelengths_weights.setter
    def wavelengths_weights(self, value):
        if "__next__" in dir(value):
            value = np.ones_like(self._wavelengths) * value
        self._wavelengths_weights = np.atleast_1d(value)

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

    def insert(self, value, key=None):
        if key is None:
            key = self.surface_pointer
        object.__setattr__(
            self, "surfaces", self.surfaces[:key] + [value] + self.surfaces[key:]
        )

    def pop(self, key=None):
        if key is None:
            key = self.surface_pointer
        object.__setattr__(
            self, "surfaces", self.surfaces[:key] + self.surfaces[key + 1 :]
        )

    def propagate(self, ray: tuple, key: int = 0, reverse: bool = False):
        propagation_array = []
        current_ray = copy(ray)
        current_ray[2] = self.surfaces[key].sag_func["sag"](
            current_ray[0], current_ray[1], **self.surfaces[key].args
        )
        current_ray.normalize(
            self.surfaces[key].material.index(
                current_ray.wavelength
                if current_ray.wavelength
                else self.wavelengths[self.reference_wavelength]
            )
        )
        coords, angle = self.get_global_vertex_coordinates()
        for suri, sur in enumerate(self.surfaces):
            current_ray[:3] += sur.args["decenter"]
            # rotate ray in local coordinates
            surface_rot = R.from_euler("xyz", angle[suri], degrees=True)
            current_ray[:3], current_ray[3:] = surface_rot.apply(
                np.c_[current_ray[:3], current_ray[3:]].T
            )
            current_ray = transfert(
                current_ray,
                lambda x, y: sur.sag_func["sag"](x, y, **sur.args),
                lambda x, y: sur.sag_func["normal"](x, y, **sur.args),
                (-1 if reverse else 1)
                * self.surfaces[suri if reverse else suri - 1].thickness,
            )
            if not current_ray:
                return None
            current_ray = refraction(
                current_ray,
                sur.sag_func["normal"](
                    ray[0],
                    ray[1],
                    **sur.args,
                ),
                n2=self.surfaces[suri - 1 if reverse else suri].material.index(
                    self.wavelengths[self.reference_wavelength]
                ),
            )
            surface_rot = R.from_euler("xyz", -angle[suri], degrees=True)
            current_ray[:3], current_ray[3:] = surface_rot.apply(
                np.c_[current_ray[:3], current_ray[3:]].T
            )
            if not current_ray:
                return None

            if not reverse:
                propagation_array.append(current_ray.vector)
        return np.array(propagation_array)[:: -1 if reverse else 1]

    # def propagate(self, ray: tuple, key: int = 0, reverse: bool = False):
    #     # if key is None:
    #     #     key = 0
    #     propagation_array = []
    #     self.wavelengths = np.atleast_1d(self.wavelengths)
    #     range_obj = (
    #         range(key - 1, 0, -1) if reverse else range(key + 1, len(self.surfaces))
    #     )
    #
    #     ray = Ray(
    #         *ray.vector[:2],
    #         self.surfaces[key].sag_func["sag"](
    #             *ray.vector[:2], **self.surfaces[key].args
    #         ),
    #         *ray.vector[3:],
    #     )
    #     ray.normalize(
    #         self.surfaces[key].material.index(
    #             self.wavelengths[self.reference_wavelength]
    #         )
    #     )
    #     if reverse:
    #         propagation_array.append(ray.vector)
    #         ray = refraction(
    #             ray,
    #             self.surfaces[key].sag_func["normal"](
    #                 ray[0],
    #                 ray[1],
    #                 **self.surfaces[key].args,
    #             ),
    #             n2=self.surfaces[key - 1].material.index(
    #                 self.wavelengths[self.reference_wavelength]
    #             ),
    #         )
    #         if not ray:
    #             return None
    #     else:
    #         propagation_array.append(ray.vector)
    #     for curr in range_obj:
    #         ray = transfert(
    #             ray,
    #             lambda x, y: self.surfaces[curr].sag_func["sag"](
    #                 x, y, **self.surfaces[curr].args
    #             ),
    #             lambda x, y: self.surfaces[curr].sag_func["normal"](
    #                 x, y, **self.surfaces[curr].args
    #             ),
    #             (-1 if reverse else 1)
    #             * self.surfaces[curr if reverse else curr - 1].thickness,
    #         )
    #         if not ray:
    #             return None
    #         if reverse:
    #             propagation_array.append(ray.vector)
    #         ray = refraction(
    #             ray,
    #             self.surfaces[curr].sag_func["normal"](
    #                 ray[0],
    #                 ray[1],
    #                 **self.surfaces[curr].args,
    #             ),
    #             n2=self.surfaces[curr - 1 if reverse else curr].material.index(
    #                 self.wavelengths[self.reference_wavelength]
    #             ),
    #         )
    #         if not ray:
    #             return None
    #
    #         if not reverse:
    #             propagation_array.append(ray.vector)
    #     if reverse:
    #         ray = transfert(
    #             ray,
    #             lambda x, y: self.surfaces[0].sag_func["sag"](
    #                 x, y, **self.surfaces[0].args
    #             ),
    #             lambda x, y: self.surfaces[0].sag_func["normal"](
    #                 x, y, **self.surfaces[0].args
    #             ),
    #             -1 * self.surfaces[0].thickness,
    #         )
    #         if not ray:
    #             return None
    #         propagation_array.append(ray.vector)
    #
    #     if propagation_array is None:
    #         raise Exception("Ray not propagated")
    #     return np.array(propagation_array)[:: -1 if reverse else 1]

    def get_global_vertex_coordinates(self):
        vertex_position_list = [np.array((0, 0, 0))]
        direction_list = [self.surfaces[0].args["rotation"]]
        for suri, sur in enumerate(self.surfaces):
            if isinstance(sur.positionning, int):
                vertex_position_list[-1] = (
                    vertex_position_list[sur.positionning] + sur.args["decenter"]
                )
                direction_list[-1] = direction_list[sur.positionning]
            else:
                vertex_position_list[-1] = (
                    vertex_position_list[-1] + sur.args["decenter"]
                )
            direction_list[-1] = direction_list[-1] + sur.args["rotation"]
            direction_list.append(direction_list[-1])
            vertex_position_list.append(
                vertex_position_list[-1]
                + R.from_euler("xyz", direction_list[-1], degrees=True).apply([0, 0, 1])
                * sur.thickness
            )
        return vertex_position_list, direction_list

    def plot(self, rays: tuple[Ray] = None, key: int = None, default_radius=1, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        coords, angle = self.get_global_vertex_coordinates()

        prevX = False
        prevDomain = False
        previndex = False
        # if rays:
        #     for ray in rays if ("__iter__" in dir(rays)) else (rays,):
        #         propagation_array = self.propagate(ray, key)
        #         ax.plot(
        #             propagation_array[:, 2] + thick,
        #             propagation_array[:, 1],
        #         )
        for suri, sur in enumerate(self.surfaces, start=0):
            domain = np.linspace(
                -default_radius if default_radius else -1,
                default_radius if default_radius else 1,
                100,
            )
            rotated = (
                R.from_euler("xyz", angle[suri], degrees=True).apply(
                    np.c_[
                        np.zeros_like(domain),
                        domain,
                        sur.sag_func["sag"](np.zeros_like(domain), domain, **sur.args),
                    ],
                )
                + coords[suri]
            )
            x1 = rotated[:, 2]
            domain = rotated[:, 1]
            if (
                (sur.material != self.surfaces[suri - 1 if suri > 0 else suri].material)
                or suri == 0
                or suri == len(self.surfaces) - 1
            ):
                index = self.surfaces[suri - 1 if suri > 0 else 0].material.index(
                    self.wavelengths[self.reference_wavelength]
                )
                if np.all(prevDomain) and index != 1:
                    if previndex == 1:
                        ax.lines[-1].remove()
                    ax.fill(
                        np.r_[x1, prevX[::-1]],
                        np.r_[domain, prevDomain[::-1]],
                        alpha=0,
                        hatch=("///" if np.all(np.isclose(angle[suri], 0)) else "|||")
                        if np.mod(suri, 2)
                        else (
                            "\\\\\\" if np.all(np.isclose(angle[suri], 0)) else "---"
                        ),
                    )
                    ax.plot(
                        np.r_[x1, prevX[::-1], x1[0]],
                        np.r_[domain, prevDomain[::-1], domain[0]],
                        color="black",
                    )
                else:
                    ax.plot(
                        rotated[:, 2],
                        rotated[:, 1],
                        "black",
                    )
                prevX = x1
                prevDomain = domain
                previndex = index
        # print(coords)
        ax.plot(
            *np.array([(x[2], x[1]) for x in coords]).T,
            "-.",
            color="black",
            linewidth=0.5,
        )

    # def plot(self, rays: tuple[Ray] = None, key: int = None, default_radius=1, ax=None):
    #     thick = np.append(
    #         0, np.cumsum(np.array([sur.thickness for sur in self.surfaces]))[:-1]
    #     )
    #     if ax is None:
    #         fig, ax = plt.subplots(1, 1)
    #     if rays:
    #         for ray in rays if ("__iter__" in dir(rays)) else (rays,):
    #             propagation_array = self.propagate(ray, key)
    #             ax.plot(
    #                 propagation_array[:, 2] + thick,
    #                 propagation_array[:, 1],
    #             )
    #     x2 = False
    #     domain2 = False
    #     index2 = False
    #     index = np.array(
    #         [
    #             sur.material.index(self.wavelengths[self.reference_wavelength])
    #             for sur in self.surfaces
    #         ]
    #     )
    #     for suri, sur in enumerate(self.surfaces):
    #         if "__iter__" in dir(default_radius):
    #             domain = np.linspace(-default_radius[suri], default_radius[suri], 100)
    #         else:
    #             domain = np.linspace(
    #                 -default_radius if default_radius else -1,
    #                 default_radius if default_radius else 1,
    #                 100,
    #             )
    #
    #         if (
    #             (sur.material != self.surfaces[suri - 1 if suri > 0 else suri].material)
    #             or suri == 0
    #             or suri == len(self.surfaces) - 1
    #         ):
    #             x1 = (
    #                 sur.sag_func["sag"](np.zeros_like(domain), domain, **sur.args)
    #                 + thick[suri]
    #             )
    #             index = self.surfaces[suri - 1 if suri > 0 else 0].material.index(
    #                 self.wavelengths[self.reference_wavelength]
    #             )
    #             if np.all(domain2) and index != 1:
    #                 if index2 == 1:
    #                     ax.lines[-1].remove()
    #                 ax.fill(
    #                     np.r_[x1, x2[::-1]],
    #                     np.r_[domain, domain2[::-1]],
    #                     alpha=0,
    #                     hatch="///" if np.mod(suri, 2) else "\\\\\\",
    #                 )
    #                 ax.plot(
    #                     np.r_[x1, x2[::-1], x1[-1]],
    #                     np.r_[domain, domain2[::-1], domain[0]],
    #                     color="black",
    #                 )
    #             else:
    #                 ax.plot(
    #                     sur.sag_func["sag"](np.zeros_like(domain), domain, **sur.args)
    #                     + thick[suri],
    #                     domain,
    #                     "black",
    #                 )
    #             x2 = x1
    #             domain2 = domain
    #             index2 = index
    #
    #     ax.plot(thick, np.zeros_like(thick), "black")
