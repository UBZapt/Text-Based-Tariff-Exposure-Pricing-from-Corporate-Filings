"""
The one place the project's chart colours are defined.

Three modules each held their own copy of these six hex values under different role names
(`points`/`fit`, `impose`/`reverse`, `series`), and one of them named another module as the source
and then copied it anyway. Four published figures drew from those copies, so a palette revision
needed three coordinated edits to keep them consistent - and nothing would have caught it if one
were missed.

Values are the dataviz reference palette's light-surface set, validated all-pairs: CVD dE 24.7 and
normal-vision dE 33.6 between the two categorical hues, both above the >=8 / >=15 floors, and both
above 3:1 contrast against the surface, so no relief encoding is obligatory. Each module keeps its
own role names - `impose`/`reverse` says something `series` does not - and builds them from here.
"""

SURFACE = "#fcfcfb"      # off-white plot ground
CATEGORICAL_1 = "#2a78d6"   # blue  - first categorical slot
CATEGORICAL_2 = "#eb6834"   # orange - second categorical slot
INK = "#0b0b0b"          # primary text and axis lines
INK_MUTED = "#52514e"     # secondary text, annotations
GRID = "#dcdcd8"         # gridlines


def roles(**mapping: str) -> dict[str, str]:
    """Build a module's role-named palette from the canonical values.

        PALETTE = palette.roles(impose="CATEGORICAL_1", reverse="CATEGORICAL_2")

    The neutrals (surface, ink, ink_muted, grid) are always included, since every chart in the
    project uses all four. An unknown value name raises rather than silently yielding None.
    """
    out = {"surface": SURFACE, "ink": INK, "ink_muted": INK_MUTED, "grid": GRID}
    for role, name in mapping.items():
        if name not in globals() or not isinstance(globals()[name], str):
            raise ValueError(f"unknown palette value {name!r} for role {role!r}")
        out[role] = globals()[name]
    return out
