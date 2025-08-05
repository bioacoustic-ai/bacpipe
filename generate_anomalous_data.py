import pandas as pd
import numpy as np
from pathlib import Path

src = "/mnt/swap/Work/Data/Amphibians/AnuranSet/annotations.csv"

annots = pd.read_csv(src)

labs, cnts = np.unique(annots.species, return_counts=True)

labs = labs[cnts < 20]

anomals = annots[annots.species.isin(labs)]

dd = {}

for lab in labs:
    dd[lab] = np.unique(anomals[anomals.species == lab].wavfilename, return_counts=True)

anomals.to_csv(Path(src).parent.joinpath("anomals.csv"))
