def parse_agf(file_name):
    catalog = {}
    nextline = False
    for line in open(file_name):
        line = line.strip()
        if nextline:
            if line.startswith("CD"):
                nextline = False
                catalog[info[1]]["B"] = line.split(" ")[1::2]
                catalog[info[1]]["C"] = line.split(" ")[2::2]
            continue
        if line.startswith("NM"):
            info = line.split(" ")
            catalog[info[1]] = {}
            catalog[info[1]]["n"] = float(info[4])
            catalog[info[1]]["vd"] = float(info[5])
            nextline = True
            continue
    return catalog


if __name__ == "__main__":
    catalog = parse_agf("../../../../Downloads/schottzemax-20220713.agf")
    print(catalog["F2"]["B"])
    print(catalog["F2"]["C"])
