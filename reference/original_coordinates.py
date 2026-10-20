import re

import pandas as pd
from pyproj import Transformer


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "ukbbs_model_data.csv"

OUTPUT_FILE = "ukbbs_model_data_with_coordinates.csv"

# Common modelling CRS for BOTH Great Britain and NI
#
# ETRS89-extended / LAEA Europe
# Units = metres
TARGET_CRS = "EPSG:3035"

# Native coordinate systems
GB_CRS = "EPSG:27700"

# Historical Irish Grid used for NI grid references
IRISH_CRS = "EPSG:29903"


# ============================================================
# COORDINATE TRANSFORMERS
# ============================================================

gb_to_common = Transformer.from_crs(
    GB_CRS,
    TARGET_CRS,
    always_xy=True
)

irish_to_common = Transformer.from_crs(
    IRISH_CRS,
    TARGET_CRS,
    always_xy=True
)


# ============================================================
# CLEAN GRID REFERENCE
# ============================================================

def clean_grid_ref(grid_ref):

    if pd.isna(grid_ref):
        return None

    # Upper case and remove spaces
    grid_ref = str(grid_ref).upper().strip()

    grid_ref = re.sub(
        r"\s+",
        "",
        grid_ref
    )

    return grid_ref


# ============================================================
# GRID REFERENCE PRECISION
#
# Examples:
#
# HU36       -> 10 km square
# HU3656     -> 1 km square
# HU365567   -> 100 m square
#
# Same principle for Irish references:
#
# J33        -> 10 km
# J3374      -> 1 km
# J338740    -> 100 m
# ============================================================

def grid_resolution(digits):

    if len(digits) % 2 != 0:
        raise ValueError(
            f"Grid reference must contain an even number "
            f"of digits: {digits}"
        )

    digits_per_axis = len(digits) // 2

    if digits_per_axis < 1 or digits_per_axis > 5:
        raise ValueError(
            f"Unsupported grid precision: {digits}"
        )

    # 1 digit per axis -> 10 km
    # 2 digits per axis -> 1 km
    # 3 digits per axis -> 100 m
    # 4 digits per axis -> 10 m
    # 5 digits per axis -> 1 m

    resolution = 10 ** (5 - digits_per_axis)

    return resolution


# ============================================================
# BRITISH NATIONAL GRID
#
# Example:
#
# HU3656
#
# HU = 100 km square
# 36 = easting inside square
# 56 = northing inside square
#
# Returns CENTRE of grid square in EPSG:27700
# ============================================================

def british_grid_to_xy(grid_ref):

    match = re.fullmatch(
        r"([A-HJ-Z]{2})(\d+)",
        grid_ref
    )

    if match is None:
        raise ValueError(
            f"Invalid British grid reference: {grid_ref}"
        )

    letters = match.group(1)
    digits = match.group(2)

    if len(digits) % 2 != 0:
        raise ValueError(
            f"Odd number of digits in {grid_ref}"
        )

    resolution = grid_resolution(digits)

    # --------------------------------------------------------
    # Convert two grid letters to 100-km square indices
    #
    # I is omitted from the National Grid alphabet
    # --------------------------------------------------------

    l1 = ord(letters[0]) - ord("A")
    l2 = ord(letters[1]) - ord("A")

    if l1 > 7:
        l1 -= 1

    if l2 > 7:
        l2 -= 1

    e100km = (
        ((l1 - 2) % 5) * 5
        + (l2 % 5)
    )

    n100km = (
        19
        - (l1 // 5) * 5
        - (l2 // 5)
    )

    # Split numeric part into easting / northing
    half = len(digits) // 2

    e_digits = digits[:half]
    n_digits = digits[half:]

    easting = (
        e100km * 100000
        + int(e_digits) * resolution
    )

    northing = (
        n100km * 100000
        + int(n_digits) * resolution
    )

    # --------------------------------------------------------
    # Move from SW corner to grid-square centre
    # --------------------------------------------------------

    easting += resolution / 2

    northing += resolution / 2

    return (
        easting,
        northing,
        resolution
    )


# ============================================================
# IRISH GRID
#
# Examples:
#
# J3374
# J 33 74
#
# Occasionally Irish references may be written:
#
# IJ3374
#
# The leading I is simply an Irish-grid identifier.
#
# Returns CENTRE of grid square in Irish Grid coordinates.
# ============================================================

def irish_grid_to_xy(grid_ref):

    # --------------------------------------------------------
    # Some datasets put an "I" before the Irish grid letter
    #
    # e.g.
    #
    # IJ3374 -> J3374
    # --------------------------------------------------------

    if re.fullmatch(
        r"I[A-HJ-Z]\d+",
        grid_ref
    ):

        grid_ref = grid_ref[1:]


    match = re.fullmatch(
        r"([A-HJ-Z])(\d+)",
        grid_ref
    )

    if match is None:
        raise ValueError(
            f"Invalid Irish grid reference: {grid_ref}"
        )

    letter = match.group(1)
    digits = match.group(2)

    if len(digits) % 2 != 0:
        raise ValueError(
            f"Odd number of digits in {grid_ref}"
        )

    resolution = grid_resolution(digits)


    # --------------------------------------------------------
    # Irish Grid 100-km layout
    #
    # North
    #
    # A B C D E
    # F G H J K
    # L M N O P
    # Q R S T U
    # V W X Y Z
    #
    # South
    #
    # Letter I is omitted.
    # --------------------------------------------------------

    irish_letters = (
        "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    )

    index = irish_letters.index(letter)

    row = index // 5
    col = index % 5

    e100km = col

    # Rows are lettered north -> south
    n100km = 4 - row


    # --------------------------------------------------------
    # Numeric part
    # --------------------------------------------------------

    half = len(digits) // 2

    e_digits = digits[:half]
    n_digits = digits[half:]

    easting = (
        e100km * 100000
        + int(e_digits) * resolution
    )

    northing = (
        n100km * 100000
        + int(n_digits) * resolution
    )


    # --------------------------------------------------------
    # Use centre of grid square
    # --------------------------------------------------------

    easting += resolution / 2

    northing += resolution / 2


    return (
        easting,
        northing,
        resolution
    )


# ============================================================
# DETERMINE GRID TYPE AND CONVERT TO COMMON CRS
# ============================================================

def convert_grid_ref(grid_ref):

    grid_ref = clean_grid_ref(grid_ref)

    if grid_ref is None:

        return pd.Series({
            "easting": None,
            "northing": None,
            "grid_system": None,
            "grid_resolution_m": None
        })


    # --------------------------------------------------------
    # IRISH GRID
    #
    # Single letter + digits:
    #
    # J3374
    #
    # or optional I prefix:
    #
    # IJ3374
    # --------------------------------------------------------

    if (
        re.fullmatch(
            r"[A-HJ-Z]\d+",
            grid_ref
        )
        or
        re.fullmatch(
            r"I[A-HJ-Z]\d+",
            grid_ref
        )
    ):

        native_easting, native_northing, resolution = (
            irish_grid_to_xy(grid_ref)
        )

        easting, northing = (
            irish_to_common.transform(
                native_easting,
                native_northing
            )
        )

        grid_system = "Irish Grid"


    # --------------------------------------------------------
    # BRITISH NATIONAL GRID
    #
    # Two letters + digits:
    #
    # HU3656
    # --------------------------------------------------------

    elif re.fullmatch(
        r"[A-HJ-Z]{2}\d+",
        grid_ref
    ):

        native_easting, native_northing, resolution = (
            british_grid_to_xy(grid_ref)
        )

        easting, northing = (
            gb_to_common.transform(
                native_easting,
                native_northing
            )
        )

        grid_system = "British National Grid"


    else:

        raise ValueError(
            f"Could not identify grid system for "
            f"'{grid_ref}'"
        )


    return pd.Series({

        # These are the columns you will use in the GAM
        "easting": easting,
        "northing": northing,

        # Useful diagnostic information
        "grid_system": grid_system,
        "grid_resolution_m": resolution

    })


# ============================================================
# LOAD BBS DATA
# ============================================================

dataset = pd.read_csv(
    INPUT_FILE
)


print(
    "Rows:",
    len(dataset)
)

print(
    "Unique grid references:",
    dataset["OS_GRID_REF"].nunique()
)


# ============================================================
# CONVERT UNIQUE GRID REFERENCES ONLY
#
# Much faster than converting the same square once for
# every species/year observation.
# ============================================================

locations = (
    dataset[
        ["OS_GRID_REF"]
    ]
    .drop_duplicates()
    .copy()
)


# ============================================================
# CONVERT COORDINATES
# ============================================================

converted = locations[
    "OS_GRID_REF"
].apply(
    convert_grid_ref
)


locations = pd.concat(
    [
        locations.reset_index(drop=True),
        converted.reset_index(drop=True)
    ],
    axis=1
)


# ============================================================
# INSPECT
# ============================================================

print("\nConverted locations:")

print(
    locations.head(20)
)


print("\nGrid systems:")

print(
    locations[
        "grid_system"
    ].value_counts(
        dropna=False
    )
)


print("\nGrid resolutions:")

print(
    locations[
        "grid_resolution_m"
    ].value_counts(
        dropna=False
    )
)


# ============================================================
# JOIN COORDINATES BACK TO FULL BBS DATA
# ============================================================

dataset = dataset.merge(

    locations,

    on="OS_GRID_REF",

    how="left"

)


# ============================================================
# CHECK FOR FAILED CONVERSIONS
# ============================================================

failed = dataset[
    dataset["easting"].isna()
    |
    dataset["northing"].isna()
]


if len(failed) > 0:

    print(
        "\nWARNING:",
        len(failed),
        "rows have missing coordinates."
    )

    print(
        failed[
            ["OS_GRID_REF"]
        ]
        .drop_duplicates()
    )

else:

    print(
        "\nAll grid references converted successfully."
    )


# ============================================================
# OPTIONAL:
# coordinates in kilometres for GAM fitting
#
# Dividing BOTH coordinates by the same value preserves
# their relative geometry.
# ============================================================

dataset["easting_km"] = (dataset["easting"] / 1000).round(4)
dataset["northing_km"] = (dataset["northing"] / 1000).round(4)

# ============================================================
# SAVE
# ============================================================

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nSaved:",
    OUTPUT_FILE
)


# ============================================================
# EXAMPLE CHECK
# ============================================================

print(
    "\nExample rows:"
)

print(
    dataset[
        [
            "OS_GRID_REF",
            "species",
            "year",
            "grid_system",
            "easting",
            "northing",
            "easting_km",
            "northing_km"
        ]
    ]
    .head(20)
)