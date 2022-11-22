import unittest
import numpy as np
from crayons import System, Surface, Ray, Material


class TestSystem(unittest.TestCase):
    def test_system(self):
        s = System()
        s.wavelengths = 587.56

    def test_add_surface(self):
        s = System()
        s.insert(1, Surface(type="sph", args={"c": -0.25}, thickness=1.5))

    def test_remove_surface(self):
        s = System()
        s.pop(1)

    # TODO Slice add/remove

    def test_propagate_chief(self):
        s = System()
        s.insert(2, Surface(type="sph", args={"c": -0.25}, thickness=1.5))
        s[2].material = Material(**{"n": 1.5, "vd": 50})
        s[3].material = Material(**{"n": 1.5, "vd": 50})
        s[0].thickness = 1.5
        s[1].thickness = 1.5
        s[2].thickness = 1.5
        s[0].args = {"c": -0.25}
        ray1 = Ray(
            *(0.00000000, -0.266052338, 0.00000000, 0.00000000, 0.17364818, 0.98480775)
        )
        ray2 = Ray(*s.propagate(ray1)[-1])
        self.assertTrue(
            np.all(np.isclose(s.propagate(ray1), s.propagate(ray2, 3, reverse=True)))
        )

    def test_propagate(self):
        s = System()
        s.insert(2, Surface(type="sph", args={"c": -0.25}, thickness=1.5))
        s[2].material = Material(**{"n": 1.5, "vd": 50})
        s[3].material = Material(**{"n": 1.5, "vd": 50})
        s[0].thickness = 1.5
        s[1].thickness = 1.5
        s[2].thickness = 1.5
        s[0].args = {"c": -0.25}
        ray = Ray(
            *(0.00000000, -0.266052338, 0.00000000, 0.00000000, 0.17364818, 0.98480775)
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    s.propagate(ray),
                    [
                        [
                            0.00000000,
                            -0.266052338,
                            -0.008857788,
                            0.00000000,
                            0.17364818,
                            0.98480775,
                        ],
                        [
                            0.00000000,
                            0.00000000,
                            0.00000000,
                            0.00000000,
                            0.17364818,
                            0.98480775,
                        ],
                        [
                            0.00000000,
                            0.26296468,
                            -0.00865316,
                            0.00000000,
                            0.20664864,
                            1.48569736,
                        ],
                        [
                            0.00000000,
                            0.47280629,
                            0.00000000,
                            0.00000000,
                            0.20664864,
                            1.48569736,
                        ],
                    ],
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
