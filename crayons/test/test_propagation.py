import unittest
import numpy as np
from crayons import propagation
from crayons import surfaces, Ray

sph = surfaces.surfaces_catalog["sph"]


class TestPropagationSph(unittest.TestCase):
    def test_find_intersection(self):
        ray = Ray(
            *(0, 0, 0, 0, 0, 1),
            587.5618,
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    propagation.find_intersection(
                        ray,
                        lambda x, y: sph["sag"](x, y, c=-0.25, rotation=[0, 0, 0]),
                        lambda x, y: sph["normal"](x, y, c=-0.25, rotation=[0, 0, 0]),
                        t=1.5,
                    ),
                    [0, 0, 0],
                )
            )
        )
        ray = Ray(
            *(0, 0, 0, 0.12635241, 0.41923115, 0.89904411),
            587.5618,
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    propagation.find_intersection(
                        ray,
                        lambda x, y: sph["sag"](x, y, c=-0.25, rotation=[0, 0, 0]),
                        lambda x, y: sph["normal"](x, y, c=-0.25, rotation=[0, 0, 0]),
                        t=1.5,
                    ),
                    [0.20212522, 0.67064166, -0.06180433],
                )
            )
        )

    def test_transfert(self):
        ray = Ray(
            *(0, 0, 0, 0.12635241, 0.41923115, 0.89904411),
            587.5618,
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    propagation.transfert(
                        ray,
                        lambda x, y: sph["sag"](x, y, c=-0.25, rotation=[0, 0, 0]),
                        lambda x, y: sph["normal"](x, y, c=-0.25, rotation=[0, 0, 0]),
                        t=1.5,
                    ).vector,
                    [
                        0.20212522,
                        0.67064166,
                        -0.06180433,
                        0.12635241,
                        0.41923115,
                        0.89904411,
                    ],
                )
            )
        )
        ray = Ray(
            *(
                -0.21081125,
                -0.69946149,
                0.00000000,
                0.12635241,
                0.41923115,
                0.89904411,
                587.5618,
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    propagation.transfert(
                        ray,
                        lambda x, y: sph["sag"](x, y, c=0.5, rotation=[0, 0, 0]),
                        lambda x, y: sph["normal"](x, y, c=0.5, rotation=[0, 0, 0]),
                        t=3,
                    ).vector,
                    [
                        0.23521425,
                        0.78042945,
                        0.17363637,
                        0.12635241,
                        0.41923115,
                        0.89904411,
                    ],
                )
            )
        )

    def test_refraction(self):
        ray = Ray(
            *(-0.21081125, -0.69946149, 0.00000000, 0.12635241, 0.41923115, 0.89904411),
            587.5618,
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    propagation.refraction(
                        propagation.transfert(
                            ray,
                            lambda x, y: sph["sag"](x, y, c=0.5, rotation=[0, 0, 0]),
                            lambda x, y: sph["normal"](x, y, c=0.5, rotation=[0, 0, 0]),
                            t=3,
                        ),
                        sph["normal"](
                            0.23521425, 0.78042945, c=0.5, rotation=[0, 0, 0]
                        ),
                        n2=np.linalg.norm(
                            (
                                0.04314450,
                                0.14315136,
                                1.54512696,
                            )
                        ),
                    ).vector,
                    [
                        0.23521425,
                        0.78042945,
                        0.17363637,
                        0.04314450,
                        0.14315136,
                        1.54512696,
                    ],
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
