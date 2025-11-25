import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from bacpipe.embedding_evaluation.label_embeddings import DefaultLabels as DL


birdnet_list = "bacpipe/bacpipe/model_specific_utils/birdnet/BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt"
MIN_FILES_PER_LABEL = 7
MIN_NR_DIFF_DAYS = 3
MAX_NR_OF_OCCURRENCES = 100
# MIN_TIME_IN_S_FOR_ANNOTATION = 1

def get_durations(df, start_col="Begin Time (s)", end_col="End Time (s)"):
    return df[end_col] - df[start_col]


def return_rare_species_df(df, start_col, end_col, label_col, file_col):
    df = df.dropna()
    labs, cnts = np.unique(df[label_col], return_counts=True)
    
    # it would be nice to include this threshold to ensure super short annotations aren't considered
    # but the problem is then it's more effort to ensure they are excluded from the noise collection
    # and if they are in the noise it defeats the whole purpose
    # df = df[get_durations(df, start_col, end_col) > MIN_TIME_IN_S_FOR_ANNOTATION]

    labs = labs[cnts < MAX_NR_OF_OCCURRENCES]
    good_lab = check_min_files_per_label(df, labs, label_col, file_col)
    
    df['start'] = df.pop(start_col)
    df['end'] = df.pop(end_col)
    df['wavfilename'] = df.pop(file_col)

    return df[df[label_col].isin(good_lab)]


def check_min_files_per_label(df, labs, label_col, file_col):
    good_lab = []

    for lab in labs:
        if len(df[df[label_col] == lab][file_col].unique()) >= MIN_FILES_PER_LABEL:
            days = np.unique(
                [DL.get_dt_filename(ff).date() 
                 for ff in df[df[label_col] == lab][file_col]]
                )
            if len(days) > MIN_NR_DIFF_DAYS:
                good_lab.append(lab)
    return good_lab


def check_for_birdnet_species(good_lab, scientific_name=False):
    bn_species = pd.read_csv(birdnet_list, header=None)
    if scientific_name:
        bn_species["Common Name"] = [
            s.split("_")[0] for s in bn_species.iloc[:, 0].values
        ]
    else:
        bn_species["Common Name"] = [
            s.split("_")[-1] for s in bn_species.iloc[:, 0].values
        ]

    remove_good_lab = []
    print(len(good_lab))
    for species in good_lab:
        if species in bn_species["Common Name"].values:
            print("removing", species)
            remove_good_lab.append(species)
        else:
            print("not in birdnet:", species)
    for to_remove in remove_good_lab:
        good_lab = good_lab[good_lab != to_remove]

    print(len(good_lab))
    return good_lab


def species_from_abbrev(df, abbrev2spec, label_col, abbrev_label_col):
    df["species"] = df[label_col]

    for spec in df[label_col].unique():
        if spec in abbrev2spec[label_col].values:
            df["species"][df[label_col].values == spec] = abbrev2spec[
                abbrev2spec[label_col] == spec
            ][abbrev_label_col].values[0]

    return df


def collect_annots(
    src, multiple_annots=False, remove_raven_duplicates=False, suffix="csv", pdkw={}
):
    if multiple_annots:
        annots = list(Path(src).rglob("*/*." + suffix))
        if "anomals" in [a.stem for a in annots]:
            annots.remove([a for a in annots if a.stem == "anomals"][0])
        annots.sort()
        df = pd.DataFrame([])
        for annot in annots:
            dff = pd.read_csv(annot, **pdkw)
            if remove_raven_duplicates:
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
        return df
    else:
        return pd.read_csv(src, **pdkw)


def make_single_label(full_df, anomals, start_col, end_col, file_col, label_col):

    anomals["start"] = anomals[start_col]
    anomals["end"] = anomals[end_col]

    df = pd.DataFrame()

    for _, row in tqdm(
        anomals.iterrows(), desc="Removing multi-labels", total=len(anomals)
    ):

        # build a temporary dataframe of annotations sharing the same file as our sound
        dff = full_df[full_df[file_col] == row[file_col]]
        dff = dff[dff[label_col] != row.species]
        dff["start"] = dff[start_col]
        dff["end"] = dff[end_col]

        if len(dff[label_col].unique()) > 0:
            # our sound is completely in an existing annotation
            within_complete = (dff.start <= row.start) & (dff.end >= row.end)

            # out sound has other annotations beginning within it
            begins_within = (
                (dff.start > row.start) & (dff.start < row.end) & (dff.end > row.end)
            )

            # our sound has other annotations ending within it
            ends_within = (
                (dff.end > row.start) & (dff.end < row.end) & (dff.start < row.start)
            )

            # out sound has other annotations beginning and ending within it
            complete_within = (dff.start >= row.start) & (dff.end <= row.end)

            if any(within_complete):
                # strict case, meaning as soon as there is a sound that
                # our vocalization is completely included in we skip
                continue

            # remove all annotations that are not in some way overlapping with our sound
            bool_combination = begins_within ^ complete_within ^ ends_within
            dff = dff[bool_combination]
            begins_within = begins_within[bool_combination]
            complete_within = complete_within[bool_combination]
            ends_within = ends_within[bool_combination]

            # isolate annotations that overlap with our sound
            new_sections = dff[complete_within]

            # extract new beginning and end times from the overlapping sections
            # if ends_within or begins_within does not exist use endpoint or startpoint
            # from existing annotation
            new_begs = [
                (
                    dff[ends_within].end.max()
                    if not str(dff[ends_within].end.max()) == "nan"
                    else row.start
                )
            ]
            new_ends = [
                (
                    dff[begins_within].start.max()
                    if not str(dff[begins_within].start.max()) == "nan"
                    else row.end
                )
            ]

            # additionally subtract the sections completely within our sound,
            # by setting their ends as our starts and their starts as our ends
            [new_begs.append(a) for a in new_sections.end]
            [new_ends.append(a) for a in new_sections.start]

            # make sure to sort the arrays to ensure that we only are left with
            # sections corresponding to only our sound without overlap
            new_begs.sort()
            new_ends.sort()

            # build it all into a new dataframe
            new_df = pd.DataFrame()
            new_df["start"] = new_begs
            new_df["end"] = new_ends
            new_df["species"] = row.species
            new_df[file_col] = row[file_col]

            # apply a minimum of 0.5 seconds vocalization
            new_df = new_df[new_df.end - new_df.start > 0.5]

            # add to pooled dataframe for entire dataset
            df = pd.concat([df, new_df], ignore_index=True)

    return df.sort_values("species")

### old functions for other datasets that turned out to not have enough anomalous sounds

def process_neotropic_katydid_sounds():  # for neotropic katydids sounds

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

    labs = labs[cnts < 100]
    labs = labs[labs != "BT"]
    labs = labs[labs != "?"]
    labs = labs[labs != "?C"]
    good_lab = []

    for lab in labs:
        if len(df[df["Analyst Tag"] == lab]["Begin File"].unique()) > 2:
            good_lab.append(lab)

    anomals = df[df["Analyst Tag"].isin(good_lab)]
    anomals["Analyst Tag"] = anomals["Analyst Tag"] = [
        abbrev2spec[sp] for sp in anomals["Analyst Tag"].values
    ]
    anomals.to_csv("data/katydid_anomals.csv")

def process_neotropical_coffee_farms():  # for neotropical coffee farms

    src = Path(
        "/media/siriussound/Extreme SSD/Recordings/terrestrial/Birds/neotropical coffee farms in Colombia and Costa Rica"
    )

    annots = "annotations_no_formatted.csv"

    df = pd.read_csv(src / annots)

    abbrev2spec = pd.read_csv(src / "species.csv")

    df["species"] = df["Species eBird Code"]

    for spec in df["Species eBird Code"].unique():
        df["species"][df["Species eBird Code"].values == spec] = abbrev2spec[
            abbrev2spec["Species eBird Code"] == spec
        ]["Common Name"].values[0]

    df = df.dropna()
    durations = df["End Time (s)"] - df["Start Time (s)"]
    labs, cnts = np.unique(df["species"], return_counts=True)
    df = df[durations > 1]
    labs = labs[cnts < MAX_NR_OF_OCCURRENCES]

    good_lab = []

    for lab in labs:
        num_of_files_per_lab = len(df[df["species"] == lab]["Filename"].unique())
        if num_of_files_per_lab > 2:
            good_lab.append(lab)
            print(lab, "->", num_of_files_per_lab)

    bn_species = pd.read_csv(birdnet_list, header=None)
    bn_species["Common Name"] = [s.split("_")[-1] for s in bn_species.iloc[:, 0].values]

    remove_good_lab = []
    print(len(good_lab))
    for species in good_lab:
        if species in bn_species["Common Name"].values:
            print("removing", species)
            remove_good_lab.append(species)
        else:
            print("not in birdnet:", species)
    for to_remove in remove_good_lab:
        good_lab.remove(to_remove)

    print(len(good_lab))

    anomals = df[df["species"].isin(good_lab)]

    anomals.to_csv("coffee_anomals.csv")

def process_dawn_chorus_cali_oreg_wash():  # for dawn chorus California, Oregon, and Washington
    src = Path(
        "/media/siriussound/Extreme SSD/Recordings/terrestrial/Birds/dawn chorus California, Oregon, and Washington/acoustic_annotations.tsv"
    )

    df = collect_annots(src, sep="\t")

    abbrev2spec = pd.read_csv(src.parent / "annotation_metadata.tsv", sep="\t")

    df = return_rare_species_df(df, "start", "end", "eBird_2021", "file")

    df = species_from_abbrev(df, abbrev2spec, "eBird_2021", "common_name")

    anomals = df[df["species"] != ("Rooster (red junglefowl)")]

    anomals.species.unique()

    # good_labels = ['']
    # no species are nearly threatened or from global south

    # anomals = anomals[anomals.species.isin(good_labels)]

    anomals.to_csv("data/dawnchorus_anomals.csv")

### datasets that have been selected, but this can of course still be expanded

def process_AnuraSet():  # for AnuranSet
    src = "/mnt/swap/Work/Data/Amphibians/AnuranSet/annotations.csv"

    df = pd.read_csv(src)

    anomals = return_rare_species_df(df, "start", "end", "species", "wavfilename")

    # anomals = make_single_label(df, anomals, "start", "end", "wavfilename", "species")

    good_labels = check_min_files_per_label(
        anomals, anomals.species.unique(), "species", "wavfilename"
    )

    anomals = anomals[anomals.species.isin(good_labels)]

    anomals.to_csv("data/anura_anomals.csv")

    a, b = np.unique(anomals.species, return_counts=True)

    print({k: v for k, v in zip(a, b)})

def process_arcticBirdSounds():  # for ArcticBirdSounds
    src = Path(
        "/media/siriussound/Extreme SSD/Recordings/terrestrial/Birds/ArcticBirdSounds/DataS1"
    )

    df = collect_annots(src, multiple_annots=True)

    abbrev2spec = pd.read_csv(src / "annotations_details.csv")

    df.overlap = df.overlap.values.astype(str)

    df = df[df.overlap == "nan"]

    df = species_from_abbrev(df, abbrev2spec, "tag", "vernacular_name")

    anomals = return_rare_species_df(df, "start", "end", "species", "file_name")

    anomals = anomals.dropna()

    # species to consider: Black-bellied Plover and White-rumped Sandpiper
    # they are the only species that are listed as vulnerable

    # anomals = make_single_label(df, anomals, "start", "end", "file_name", "species")

    good_labels = check_min_files_per_label(
        anomals, anomals.species.unique(), "species", "wavfilename"
    )

    # all other species are quite common actually
    good_labels = ["Black-bellied Plover", "White-rumped Sandpiper"]

    anomals = anomals[anomals.species.isin(good_labels)]

    anomals.to_csv("data/arctic_anomals.csv")

    a, b = np.unique(anomals.species, return_counts=True)

    print({k: v for k, v in zip(a, b)})

def process_WABAD():  # for WABAD
    src = Path(
        "/media/siriussound/Extreme SSD/Recordings/terrestrial/Birds/WABAD/Pooled annotations.csv"
    )

    # df = collect_annots(src.parent, suffix='txt', sep=';', multiple_annots=True)

    df = pd.read_csv(src, sep=";", encoding="iso8859_2", decimal=",")

    anomals = return_rare_species_df(df, "Begin Time (s)", "End Time (s)", "Species", "Recording")

    good_labels = check_for_birdnet_species(
        anomals.Species.unique(), scientific_name=True
    )

    anomals = anomals[anomals.Species.isin(good_labels)]

    anomals["species"] = anomals.pop("Species")

    # anomals = make_single_label(
    #     df, anomals, "Begin Time (s)", "End Time (s)", "Recording", "Species"
    # )

    good_labels = check_min_files_per_label(
        anomals, anomals.species.unique(), "species", "wavfilename"
    )

    anomals = anomals[anomals.species.isin(good_labels)]

    # reason for removal is that they occurr in multiple files but only on the same day in the span of 2 hrs
    remove_species = [
        "Anthracothorax dominicus",
        "Cnemathraupis eximia",
        "Curruca cantillans",
        "Emberiza goslingi",
    ]

    anomals = anomals[~anomals.species.isin(remove_species)]

    anomals.to_csv("data/wabad_anomals.csv")

    a, b = np.unique(anomals.species, return_counts=True)

    print({k: v for k, v in zip(a, b)})





############################## 
######### EXECUTE ############
############################## 


# process_neotropic_katydid_sounds()
# process_neotropical_coffee_farms()
# process_dawn_chorus_cali_oreg_wash()
process_WABAD()
process_arcticBirdSounds()
process_AnuraSet()