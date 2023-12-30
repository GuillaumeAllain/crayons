from .system import System, Surface, Material
from pathlib import Path
import pyparsing as pp
from re import match
import numpy as np

__accepted_extensions = [".seq"]

codev_parser = pp.ZeroOrMore(
    pp.Word(pp.alphanums) + pp.Group(pp.ZeroOrMore(pp.Word(pp.printables)))
)


def file_import(filename):
    assert (
        Path(filename).suffix in __accepted_extensions
    ), f"File extension not supported: {Path(filename).suffix}"
    # init system

    with open(Path(filename)) as f:
        raw_file = f.read()
    raw_file = raw_file.replace(";", "\n")
    system = System()
    locals = {"RDM": True, "surface_pointer": 0}
    for line in raw_file.split("\n"):
        command_line = codev_parser.parse_string(line)
        if len(command_line) > 0:
            if command_line[0].upper() in codev_commands.keys():
                codev_commands[command_line[0].upper()](
                    locals, system, command_line[1].asList()
                )
    return system


codev_commands = {
    "LEN": lambda locals, system, args: (
        system.__init__(),
        system.pop(1),
        setattr(system, "wavelengths", list()),
    ),
    "WL": lambda locals, system, args: (
        setattr(system, "wavelengths", [float(x) for x in args])
    ),
    "S": lambda locals, system, args: (
        locals.update({"surface_pointer": locals["surface_pointer"] + 1}),
        system.insert(
            Surface(
                type="sph",
                thickness=float(args[1]) if len(args) > 1 else 0,
                args={
                    "c": 0
                    if float(args[0]) == 0
                    else 1 / float(args[0])
                    if locals["RDM"]
                    else float(args[0])
                    if len(args) > 0
                    else 0
                },
                material=parse_codev_material(args[2])
                if len(args) > 2
                else Material(name="air"),
            ),
            locals["surface_pointer"],
        ),
    ),
    "YDE": lambda locals, system, args: (
        system.surfaces[locals["surface_pointer"]].args.update(
            {
                "decenter": np.array(
                    [
                        system.surfaces[locals["surface_pointer"]].args["decenter"][0]
                        if "decenter"
                        in system.surfaces[locals["surface_pointer"]].args.keys()
                        else 0,
                        float(args[0]),
                        system.surfaces[locals["surface_pointer"]].args["decenter"][2]
                        if "decenter"
                        in system.surfaces[locals["surface_pointer"]].args.keys()
                        else 0,
                    ]
                ),
            },
        ),
    ),
    "XDE": lambda locals, system, args: (
        system.surfaces[locals["surface_pointer"]].args.update(
            {
                "decenter": np.array(
                    [
                        float(args[0]),
                        system.surfaces[locals["surface_pointer"]].args["decenter"][1]
                        if "decenter"
                        in system.surfaces[locals["surface_pointer"]].args.keys()
                        else 0,
                        system.surfaces[locals["surface_pointer"]].args["decenter"][2]
                        if "decenter"
                        in system.surfaces[locals["surface_pointer"]].args.keys()
                        else 0,
                    ]
                ),
            },
        ),
    ),
    "ZDE": lambda locals, system, args: (
        system.surfaces[locals["surface_pointer"]].args.update(
            {
                "decenter": np.array(
                    [
                        system.surfaces[locals["surface_pointer"]].args["decenter"][0]
                        if "decenter"
                        in system.surfaces[locals["surface_pointer"]].args.keys()
                        else 0,
                        system.surfaces[locals["surface_pointer"]].args["decenter"][1]
                        if "decenter"
                        in system.surfaces[locals["surface_pointer"]].args.keys()
                        else 0,
                        float(args[0]),
                    ]
                ),
            },
        ),
    ),
    "BDE": lambda locals, system, args: (
        system.surfaces[locals["surface_pointer"]].args.update(
            {
                "rotation": np.array(
                    [
                        system.surfaces[locals["surface_pointer"]].args["rotation"][0]
                        if "rotation"
                        in system.surfaces[locals["surface_pointer"]].args.keys()
                        else 0,
                        float(args[0]),
                        system.surfaces[locals["surface_pointer"]].args["rotation"][2]
                        if "rotation"
                        in system.surfaces[locals["surface_pointer"]].args.keys()
                        else 0,
                    ]
                ),
            },
        ),
    ),
    "ADE": lambda locals, system, args: (
        system.surfaces[locals["surface_pointer"]].args.update(
            {
                "rotation": np.array(
                    [
                        float(args[0]),
                        system.surfaces[locals["surface_pointer"]].args["rotation"][1]
                        if "rotation"
                        in system.surfaces[locals["surface_pointer"]].args.keys()
                        else 0,
                        system.surfaces[locals["surface_pointer"]].args["rotation"][2]
                        if "rotation"
                        in system.surfaces[locals["surface_pointer"]].args.keys()
                        else 0,
                    ]
                ),
            },
        ),
    ),
    "CDE": lambda locals, system, args: (
        system.surfaces[locals["surface_pointer"]].args.update(
            {
                "rotation": np.array(
                    [
                        system.surfaces[locals["surface_pointer"]].args["rotation"][0]
                        if "rotation"
                        in system.surfaces[locals["surface_pointer"]].args.keys()
                        else 0,
                        system.surfaces[locals["surface_pointer"]].args["rotation"][1]
                        if "rotation"
                        in system.surfaces[locals["surface_pointer"]].args.keys()
                        else 0,
                        float(args[0]),
                    ]
                ),
            },
        ),
    ),
    "STO": lambda locals, system, args: (
        setattr(system, "stop", args[0] if len(args) > 0 else locals["surface_pointer"])
    ),
    "RDM": lambda locals, system, args: (
        locals.update(
            {
                "RDM": True
                if len(args) == 0
                else True
                if args[0].upper in "YES"
                else False,
            }
        )
    ),
    "GLB": lambda locals, system, args: (
        setattr(
            system.surfaces[locals["surface_pointer"]], "positionning", int(args[0][1:])
        ),
    ),
    "CIR": lambda locals, system, args: (
        system.surfaces[locals["surface_pointer"]]
        .args["aperture"][0]
        .update({"type": "circular", "cir": float(args[-1])})
        if len(system.surfaces[locals["surface_pointer"]].args["aperture"]) > 1
        else system.surfaces[locals["surface_pointer"]]
        .args["aperture"]
        .append({"type": "circular", "cir": float(args[-1])})
        if "EDG" not in [x.upper for x in args]
        else None
    ),
}


def parse_codev_material(input):
    if input == "AIR":
        return Material(name="air")
    elif match("^[0-9]+[.][0-9]+$", input):
        return Material(code=input.replace(".", ":"))
    elif match("^[A-Z0-9]+[_][A-Z]+$", input.upper()):
        splinput = input.split("_")
        return Material(name=splinput[0].upper(), catalog=splinput[1].upper())

    # elif match("^[A-Z0-9]+[_][A-Z]+$", input.upper()):
    #     splinput = input.split("_")
    #     match splinput[1].upper():
    #         case "SCHOTT":
    #             splinput[0] = (
    #                 splinput[0][:1] + "-" + splinput[0][1:]
    #                 if splinput[0].startswith("N")
    #                 else splinput[0]
    #             )
    #         case "CDGM":
    #             splinput[0] = (
    #                 splinput[0][:1] + "-" + splinput[0][1:]
    #                 if splinput[0].startswith("H")
    #                 else splinput[0]
    #             )
    #         case "OHARA":
    #             splinput[0] = (
    #                 splinput[0][:1] + "-" + splinput[0][1:]
    #                 if splinput[0].startswith("S")
    #                 else splinput[0]
    #             )

    # return Material(name=splinput[0].upper(), catalog=splinput[1].upper())
    else:
        raise Exception(f"Material {input} not recognized")
