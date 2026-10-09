"""
The one place the project's chart colours are defined.

Three modules each held their own copy of these hex values under different role names
(`points`/`fit`, `impose`/`reverse`, `series`), and one of them named another module as the source
and then copied it anyway. Four published figures drew from those copies, so a palette revision
needed three coordinated edits to keep them consistent - and nothing would have caught it if one
were missed.

The values are muted rather than saturated: a desaturated dark blue, a mid grey and a desaturated
dark green on a white ground. The earlier set was R base graphics' primaries, which print harshly
and, at line weight, read as brighter than the data they carry. They are chosen for a printed
page, not for measured colour-vision separation: the blue/grey pair separates mainly on lightness
rather than hue, so every figure using more than one categorical slot also carries a hatch, a dash
pattern or a direct label, and none relies on hue alone.

`DIVERGE_WARM` is the warm end of the correlation grids' fixed [-1, 1] scale and is never a data
series. `GRID` is the pale fill of a shaded span rather than a gridline - the reference style
draws no gridlines at all. There is no colour for the reference line at zero: it is drawn solid in
`INK`, which is `chartstyle.zero_line`'s business, not a palette slot.

Each module keeps its own role names - `impose`/`reverse` says something `series` does not - and
builds them from here. `chartstyle` consumes the neutrals for the axis furniture.
"""

SURFACE = "#ffffff"          # white plot ground
CATEGORICAL_1 = "#3d5a80"    # muted dark blue  - first categorical slot
CATEGORICAL_2 = "#9a9a9a"    # mid grey         - second categorical slot
CATEGORICAL_3 = "#4f7a5b"    # muted dark green - third categorical slot
DIVERGE_WARM = "#b2474d"     # muted red; the warm end of a correlation scale only
INK = "#000000"              # text, axis lines, tick marks and the reference line at zero
INK_MUTED = "#404040"        # secondary text and secondary reference lines
GRID = "#d9d9d9"             # pale fill for shaded spans and bands; never a gridline


def roles(**mapping: str) -> dict[str, str]:
    """Build a module's role-named palette from the canonical values.

        PALETTE = palette.roles(impose="CATEGORICAL_1", reverse="CATEGORICAL_2")

    The neutrals (surface, ink, ink_muted, grid) are always included, since every chart in the
    project uses them. An unknown value name raises rather than silently yielding None.
    """
    out = {"surface": SURFACE, "ink": INK, "ink_muted": INK_MUTED, "grid": GRID}
    for role, name in mapping.items():
        if name not in globals() or not isinstance(globals()[name], str):
            raise ValueError(f"unknown palette value {name!r} for role {role!r}")
        out[role] = globals()[name]
    return out
