import os

DATA_DIR = os.path.normpath(os.path.join(os.path.abspath(".."), "data"))

CLASSNAMES = [
    "cel",
    "cla",
    "flu",
    "gac",
    "gel",
    "org",
    "pia",
    "sax",
    "tru",
    "vio",
    "voi",
]
CLASS_COLORS = [
    "#e6194B",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabebe",
    "#469990",
]

FEATURE_DIR = os.path.normpath(os.path.join(os.getcwd(), "sample_features"))

SAMPLE_DIR = os.path.normpath(os.path.join(os.getcwd(), "sample_data"))
