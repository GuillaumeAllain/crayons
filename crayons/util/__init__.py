import xml.etree.ElementTree as ET


def parse_agf(file_name):
    catalog = {}
    nextline = False
    for line in open(file_name):
        line = line.strip()
        if nextline:
            if line.startswith("CD"):
                nextline = False
                catalog[info[1]]["B"] = tuple([float(x) for x in line.split(" ")[1::2]])
                catalog[info[1]]["C"] = tuple([float(x) for x in line.split(" ")[2::2]])
            continue
        if line.startswith("NM"):
            info = line.split(" ")
            catalog[info[1]] = {}
            catalog[info[1]]["n"] = float(info[4])
            catalog[info[1]]["vd"] = float(info[5])
            nextline = True
            continue
    return catalog


def parse_xml(file_name):
    catalog = {}
    tree = ET.parse(file_name)
    for child in tree.getroot()[-1].iter("Glass"):
        coef = [
            float(x.text)
            for x in child.find("DispersionCoefficients").iter()
            if x.tag == "Coefficient"
        ]
        name = child.find("GlassName").text
        catalog[name] = {}
        catalog[name]["B"] = tuple(coef[0::2])
        catalog[name]["C"] = tuple(coef[1::2])
    return catalog


if __name__ == "__main__":
    from pathlib import Path

    special = parse_xml(f"{Path(__file__).parent}/../catalogs/SPECIAL.xml")
    print(special["ACRYLIC"]["B"])

    # catalog = {}
    # catalog["SCHOTT"] = parse_agf(f"{Path(__file__).parent}/../catalogs/SCHOTT.agf")
    # print(catalog["SCHOTT"]["N-BK7"]["B"])
    # print(catalog["SCHOTT"]["F2"]["C"])
