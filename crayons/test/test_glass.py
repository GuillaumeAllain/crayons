from crayons.util import parse_agf
import unittest


class TestCatalog(unittest.TestCase):
    def test_parse_agf(self):
        # catalog = parse_agf('tinyrayopt/data/catalog.agf')
        catalog = parse_agf("~/Download/schottzemax-20220713.agf")
        # self.assertEqual(catalog["BK7"]["n"], 1.5168)
        # self.assertEqual(catalog['BK7']['k'], 0.0000)


if __name__ == "__main__":
    # unittest.main()
    parse_agf("~Download/schottzemax-20220713.agf")
