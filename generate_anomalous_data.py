import pandas as pd
import numpy as np
from pathlib import Path


if False:  # for AnuranSet
    src = "/mnt/swap/Work/Data/Amphibians/AnuranSet/annotations.csv"

    annots = pd.read_csv(src)

    labs, cnts = np.unique(annots.species, return_counts=True)

    labs = labs[cnts < 20]

    anomals = annots[annots.species.isin(labs)]

    dd = {}

    for lab in labs:
        dd[lab] = np.unique(
            anomals[anomals.species == lab].wavfilename, return_counts=True
        )

    anomals.to_csv(Path(src).parent.joinpath("anomals.csv"))

if True:  # for neotropic katydids sounds

    srcs = "/media/siriussound/Extreme SSD/Recordings/terrestrial/Insects/neotropic katydids sounds/7591959"
    annots = list(Path(srcs).glob("*.txt"))
    annots.sort()

    df = pd.DataFrame([])
    for annot in annots:
        dff = pd.read_csv(annot, sep="\t")
        dff = dff[dff.View == "Spectrogram 1"]
        dff = dff.loc[
            :,
            [
                "Begin Time (s)",
                "End Time (s)",
                "Begin File",
                "High Freq (Hz)",
                "Analyst Tag",
            ],
        ].sort_values(by="Begin Time (s)")

        df = pd.concat([df, dff], ignore_index=True)
        print("added ", annot)

    abbrev2spec = {
        "AMAJOR": "Acantheremus major",
        "ACURV": "Acanthodis curvidens",
        "AGFEST": "Agraecia festae",
        "ACOLO": "Anapolisa colossea",
        "AD": "Anaulacomera darwinii",
        "AFURC": "Anaulacomera furcata",
        "AS": "Anaulacomera spatulata",
        "AW or HS": "Anaulacomera wallace/Hetaira sp.",
        "AW/HS": "Anaulacomera wallace/Hetaira sp.",
        "CD": "Chloroscirtus discocercus",
        "CM": "Ceraia mytra",
        "DG": "Docidocercus gigliotosi",
        "DL": "Dolichocercus latipennis",
        "ED": "Ectemna dumicola",
        "EA": "Euceraia atryx",
        "EI": "Euceraia insignis",
        "EL": "Erioloides longinoi",
        "HI": "Hyperphrona irregularis",
        "IP": "Ischnomela pulchripennis",
        "MC": "Microcentrum championi",
        "MB": "Montezumina bradleyi",
        "PQ": "Phylloptera quinquemaculata",
        "PT": "Pristonotus tuberosus",
        "TS": "Thamnobates subfalcata",
        "VB": "Viadana brunneri",
    }
    df = df.dropna()
    durations = df["End Time (s)"] - df["Begin Time (s)"]
    labs, cnts = np.unique(df["Analyst Tag"], return_counts=True)
    df = df[durations > 1]

    labs = labs[cnts < 20]
    labs = labs[labs != "BT"]
    labs = labs[labs != "?"]
    labs = labs[labs != "?C"]

    anomals = df[df["Analyst Tag"].isin(labs)]
    anomals["Analyst Tag"] = anomals["Analyst Tag"] = [
        abbrev2spec[sp] for sp in anomals["Analyst Tag"].values
    ]
    anomals.to_csv(Path(srcs).parent.joinpath("anomals.csv"))
