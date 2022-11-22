import unittest
import numpy as np
from crayons import surfaces, Ray

sph = surfaces.surfaces_catalog["sph"]


class TestRay(unittest.TestCase):
    def test_ray_getitem(self):
        R = Ray(0, 1, 2, 3, 4, 4)
        self.assertTrue(np.all(np.equal(R[0::2], np.array([0, 2, 4]))))


class TestSph(unittest.TestCase):
    def test_sag(self):
        self.assertTrue(np.array_equal(sph["sag"](0, 0), 0))
        self.assertTrue(np.array_equal(sph["sag"](1, 1), 0))
        self.assertAlmostEqual(sph["sag"](1, 1, c=0.5), 0.5857864376269)
        self.assertAlmostEqual(sph["sag"](1, 0.25, c=0.75), 0.48759236957565)
        self.assertAlmostEqual(sph["sag"](0, 0.05, c=0.05), 0.62500097657e-4)

    def test_sag_array(self):
        self.assertTrue(np.all(np.isclose(sph["sag"]([0, 0], [0, 0]), [0, 0])))
        self.assertTrue(
            np.all(
                np.isclose(
                    sph["sag"]([1, 0.75, 0, 0.25], [1, 0.75, 0.25, 1.25], c=0.75),
                    np.array(
                        [np.nan, 0.52538669043061, 0.02364719620819, 0.94246535334805]
                    ),
                    equal_nan=True,
                )
            )
        )

    def test_sag_normal(self):
        self.assertTrue(np.array_equal(sph["normal"](0, 0), [0, 0, -1]))
        self.assertTrue(np.array_equal(sph["normal"](1, 1), [0, 0, -1]))
        self.assertTrue(
            np.all(np.isclose(sph["normal"](0, 0.26766, c=0.5), [0, 0.13383, -0.99100]))
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    sph["normal"](0.38204497, 0.66486670, c=-0.25),
                    [-0.09551124, -0.16621668, -0.98145281],
                )
            )
        )

    def test_sag_normal_array(self):
        self.assertTrue(
            np.all(
                np.isclose(sph["normal"]([0, 0], [0, 0]), [[0, 0], [0, 0], [-1, -1]])
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    sph["normal"](
                        [0, 0.64960168, 0.20212522],
                        [0, 0.64960168, 0.67064166],
                        c=-0.25,
                    ),
                    np.array(
                        [
                            [0, -0.16240042, -0.05053131],
                            [0, -0.16240042, -0.1676604],
                            [-1, -0.97326883, -0.98454892],
                        ]
                    ),
                    equal_nan=True,
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
