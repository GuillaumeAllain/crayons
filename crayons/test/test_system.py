import unittest
import numpy as np
from crayons import System, Surface, Ray, Material


class TestSystem(unittest.TestCase):
    def test_system(self):
        s = System()
        s.wavelengths = 587.56

    def test_add_surface(self):
        s1 = System()
        s1.insert(Surface(type="sph", args={"c": -0.25}, thickness=1.5), 1)
        s1.insert(Surface(type="sph", args={"c": -0.21}, thickness=1.5))
        s1.surface_pointer = 1
        s1.insert(Surface(type="sph", args={"c": -0.23}, thickness=1.5))
        s2 = System()
        s2.insert(Surface(type="sph", args={"c": -0.25}, thickness=1.5), 1)
        s2.insert(Surface(type="sph", args={"c": -0.21}, thickness=1.5), 0)
        s2.insert(Surface(type="sph", args={"c": -0.23}, thickness=1.5), 1)
        # self.assertEqual(np.all(s2.surfaces == s1.surfaces))

    def test_remove_surface(self):
        s = System()
        s.pop(1)

    def test_propagate_reverse1(self):
        s = System()
        s.insert(Surface(type="sph", args={"c": -0.25}, thickness=1.5), 2)
        s[2].material = Material(**{"n": 1.5, "vd": 50})
        s[3].material = Material(**{"n": 1.5, "vd": 50})
        s[0].thickness = 1.5
        s[1].thickness = 1.5
        s[2].thickness = 1.5
        s[0].args.update({"c": -0.25})
        ray1 = Ray(
            *(0.00000000, -0.266052338, 0.00000000, 0.00000000, 0.17364818, 0.98480775),
            wavelength=587.56
        )
        ray2 = Ray(*s.propagate(ray1)[-1], wavelength=587.56)
        self.assertTrue(
            np.all(np.isclose(s.propagate(ray1), s.propagate(ray2, 3, reverse=True)))
        )

    def test_propagate_chief2(self):
        s = System()
        s.insert(Surface(type="sph", args={"c": -0.25}, thickness=1.5), 2)
        s[2].material = Material(**{"n": 1.5, "vd": 50})
        s[3].material = Material(**{"n": 1.5, "vd": 50})
        s[0].thickness = 1.5
        s[1].thickness = 1.5
        s[2].thickness = 1.5
        s[0].args.update({"c": -0.25})
        s.stop = 1
        s.pop(1)
        ray1 = Ray(
            *(0.00000000, -0.266052338, 0.00000000, 0.00000000, 0.17364818, 0.98480775),
            wavelength=587.56
        )
        ray2 = Ray(*s.propagate(ray1)[-1], wavelength=587.56)
        self.assertTrue(
            np.all(
                np.isclose(
                    s.propagate(ray1),
                    s.propagate(ray2, len(s.surfaces) - 1, reverse=True),
                )
            )
        )

    def test_propagate_chief3(self):
        s = System(
            surfaces=[
                Surface("sph", thickness=6, args={"c": 0}),
                Surface(
                    "sph",
                    thickness=1,
                    args={"c": 0.01},
                    material=Material(n=1.5, vd=50),
                ),
                Surface(
                    "sph", thickness=1, args={"c": 0}, material=Material(n=1.5, vd=50)
                ),
            ],
            stop=1,
        )
        ray1 = Ray(
            *(
                0.000000000,
                -0.524931981,
                0.000000000,
                0.000000000,
                0.168382942,
                0.985721657,
            ),
            wavelength=587.56
        )
        ray2 = Ray(*s.propagate(ray1)[-1], wavelength=587.56)
        self.assertTrue(
            np.all(
                np.isclose(
                    s.propagate(ray1),
                    s.propagate(ray2, len(s.surfaces) - 1, reverse=True),
                )
            ),
        )

    def test_propagate(self):
        s = System()
        s.insert(Surface(type="sph", args={"c": -0.25}, thickness=1.5), 2)
        s[2].material = Material(**{"n": 1.5, "vd": 50})
        s[3].material = Material(**{"n": 1.5, "vd": 50})
        s[0].thickness = 1.5
        s[1].thickness = 1.5
        s[2].thickness = 1.5
        s[0].args.update({"c": -0.25})
        ray = Ray(
            *(0.00000000, -0.266052338, 0.00000000, 0.00000000, 0.17364818, 0.98480775),
            wavelength=587.56
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


class TestPlotSystem(unittest.TestCase):
    def test_plot_system(self):
        # s = System()
        # s.insert(2, Surface(type="sph", args={"c": 0.25}, thickness=1.5))
        # s.insert(3, Surface(type="sph", args={"c": -0.25}, thickness=1.5))
        # s[1].material = Material(**{"n": 0.7, "vd": 50})
        # s[2].material = Material(**{"n": 1.7, "vd": 50})
        # s[0].thickness = 1.5
        # s[1].thickness = 1.5
        # s[2].thickness = 1.5
        # s[0].thickness = 1
        # s[1].thickness = 1
        # s[1].args["incline"] = (0.0, 0.05, 1)
        # s[1].material = Material(n=1.5, vd=50)
        # s[2].material = Material(n=1.5, vd=50)
        # s.plot((Ray(0, 0, 0, 0, 0, 1),))
        # s.show()
        # raise NotImplementedError
        pass


if __name__ == "__main__":
    unittest.main()
