import bacpipe
import numpy as np
import re
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger("bacpipe")
from sklearn.metrics import classification_report, average_precision_score

    
def clean_string(s):
    return re.sub(r"[-\s]", "", s).lower()


def normalize_name(s):
    # Lowercase, standardize grey -> gray, and remove ALL non-alphanumeric chars
    # Handles hyphens, spaces, slashes (/), apostrophes, etc.
    s = s.lower().replace("grey", "gray")
    return re.sub(r"[^a-z0-9]", "", s)

def associate_labels_to_eBird_Codes(gt_species_cols, gt_without_metadata):
    logger.info(
        "\nNo species found on first attempt, therefore trying to see if "
        "annotated species are eBird Codes and converting them to common name."
    )
    # check if in eBird Codes
    ebird_path = Path('bacpipe/embedding_evaluation/eBird_taxonomy_v2025-4.csv')
    ebird_df = pd.read_csv(ebird_path)
    gt_ebird2common = {}
    for idx, gt_label in enumerate(gt_species_cols):
        if gt_label in ebird_df.SPECIES_CODE.values:
            gt_ebird2common[gt_label] = (
                idx, 
                ebird_df[
                    ebird_df.SPECIES_CODE == gt_label
                    ].PRIMARY_COM_NAME.values[0]
                )
    if len(gt_ebird2common) > 0:
        for idx, new_label in gt_ebird2common.values():
            gt_species_cols[idx] = new_label
            
        # EDIT THE GROUND TRUTH DATAFRAME COLUMNS
        gt_without_metadata.rename(columns={k: v[1] for k, v in gt_ebird2common.items()}, inplace=True)
        
        logger.info(
            "\nThe following species were found and converted from eBird Codes "
            f"to common names: \n"
        )
        for key, (idx, new_label) in gt_ebird2common.items():
            logger.info(f"{key} --> {new_label}")
    return gt_species_cols, gt_without_metadata
            
def associate_labels_regardless_of_puctuation(
    label2idx, gt_without_metadata, found, not_found
    ):
    logger.info(
        f"\nSpecies found in ground truth but NOT exactly in predictions: {not_found}"
    )
    l2i_regex = {clean_string(lbl): lbl for lbl in label2idx.keys()}
    still_not_found = []

    for label in not_found:
        cleaned = clean_string(label)
        if cleaned in l2i_regex:
            matched_label = l2i_regex[cleaned]
            
            # EDIT THE GROUND TRUTH DATAFRAME COLUMNS
            gt_without_metadata.rename(columns={label: matched_label}, inplace=True)
            
            logger.info(
                f"With regex we matched ground truth '{label}' to prediction '{matched_label}'"
            )
            found.append(label)
        else:
            still_not_found.append(label)

    not_found = still_not_found
    logger.info(f"Remaining unmatched species: {not_found}")
    return gt_without_metadata

def associate_labels_regardless_of_spelling_and_substrings(
    label2idx,
    found,
    gt_without_metadata,
    not_found
    ):
    logger.info(
        f"\nNext try: Flexible matching (grey/gray, slashes, and unique substrings) for: {not_found}"
    )


    # Map normalized prediction labels back to their true label
    # e.g., 'easternwesternwarblingvireo' -> 'Eastern/Western Warbling Vireo'
    norm2pred = {normalize_name(lbl): lbl for lbl in label2idx.keys()}
    
    still_not_found = []
    # Keep track of column renames needed for gt_without_metadata
    rename_map = {} 

    for gt_label in not_found:
        norm_gt = normalize_name(gt_label)
        
        # 1. Exact Normalized Match (Catches grey/gray and slash/punctuation differences)
        if norm_gt in norm2pred:
            matched_label = norm2pred[norm_gt]
            logger.info(
                f"\nMatched (via grey/gray & punctuation normalization) '{gt_label}' --> '{matched_label}'"
            )
            found.append(gt_label)
            rename_map[gt_label] = matched_label
            continue
        
        # 2. Substring Match (Catches 'Warbling Vireo' in 'Eastern/Western Warbling Vireo')
        # We check both directions: gt inside pred, OR pred inside gt.
        # We enforce min length >= 5 to prevent tiny generic strings from matching wildly.
        candidates = []
        if len(norm_gt) >= 5:
            for norm_pred, orig_pred in norm2pred.items():
                if norm_gt in norm_pred or norm_pred in norm_gt:
                    candidates.append(orig_pred)
        
        # SAFEGUARD: Only associate if there is exactly ONE unambiguous match
        if len(candidates) == 1:
            matched_label = candidates[0]
            logger.info(
                f"\nMatched (via unique substring) '{gt_label}' --> '{matched_label}'"
            )
            found.append(gt_label)
            rename_map[gt_label] = matched_label
        elif len(candidates) > 1:
            logger.warning(
                f"\nAMBIGUOUS MATCH: Ground truth '{gt_label}' matches multiple model classes: {candidates}. "
                "Skipping to prevent misassociation!"
            )
            still_not_found.append(gt_label)
        else:
            still_not_found.append(gt_label)

    # EDIT THE GROUND TRUTH DATAFRAME COLUMNS
    if rename_map:
        gt_without_metadata.rename(columns=rename_map, inplace=True)
    # ----------------------------------------

    # IMPORTANT: Rename columns in your dataframe so reindex() and alignment work later!
    # if rename_map and 'gt_without_metadata' in locals():
    #     gt_without_metadata.rename(columns=rename_map, inplace=True)

    not_found = still_not_found
    logger.info(f"\nRemaining unmatched species after flexible matching: {not_found}")
    
    return gt_without_metadata

def associate_ground_truth_and_prediction_labels(
    gt_species_cols, 
    label2idx, 
    gt_without_metadata
    ):
    # Find exact matching classes
    found = [label for label in gt_species_cols if label in label2idx]
    not_found = [label for label in gt_species_cols if label not in label2idx]
    
    if len(found) == 0:
            gt_species_cols, gt_without_metadata = associate_labels_to_eBird_Codes(
                gt_species_cols, gt_without_metadata
                )
                    
            found = [label for label in gt_species_cols if label in label2idx]
            not_found = [label for label in gt_species_cols if label not in label2idx]

    # Fallback to Regex matching for missing classes
    if not_found:
        gt_without_metadata = associate_labels_regardless_of_puctuation(
            label2idx, gt_without_metadata, found, not_found
            )
        found = [col for col in gt_without_metadata.columns if col in label2idx]
        not_found = [col for col in gt_without_metadata.columns if col not in label2idx]
    
    # accommodate differences (grey/gray, punctuation/slashes, and unique substrings)
    if not_found:
        gt_without_metadata = associate_labels_regardless_of_spelling_and_substrings(
            label2idx,
            found,
            gt_without_metadata,
            not_found
        )
        not_found = [col for col in gt_without_metadata.columns if col not in label2idx]

    if not found:
        logger.error(
            "\nNo ground truth classes have been found in the predictions.\n"
        )
        return {
            "error": "No ground truth classes have been found in the predictions."
        }
        
        
    logger.info(
        "\nThe following species were found in the ground truth and the predictions:"
    )
    for label in found:
        logger.info(f" - {label}")

    # --- Matrix Generation & Alignment (Condensed) ---
    # 1. Align ground truth to label2idx columns, filling missing ones with 0
    gt_aligned = gt_without_metadata.reindex(
        columns=label2idx.keys(), fill_value=0
    )

    # 2. Get shared labels and their corresponding integer indices
    shared_labels = [
        lbl for lbl in label2idx.keys() if lbl in gt_without_metadata.columns
    ]
    shared_indices = [label2idx[lbl] for lbl in shared_labels]
    return gt_aligned, shared_labels, shared_indices, not_found



def benchmark(
    model,
    dataset,
    annotations_file=None,
    CustomModel=None,
    check_if_already_processed=True,
    min_annotation_length=0.01,
    overwrite=True,
    **kwargs,
):
    """
    Benchmark a model's classifier performance for a dataset.
    The dataset requires an annotation file that is located in
    the root directory of the dataset. This annotation file has
    needs to have the column names: `start`, `end`,
    `audiofilename`, `label:species` so that the ground truth
    can be extracted.
    Ground truth is mapped to the timestamps so that predictions
    and ground_truth have the same shape.
    If predictions have already been produced this function runs
    very quickly as it uses the saved data.

    Finally the sklearn.metrics.classification_report function
    is used to quantify the performance. The results are printed
    as a report and returned as a dictionary.
    This function expects a threshold. Threshold-independent
    performance evaluation is currently not supported.

    Parameters
    ----------
    model : string
        model name
    dataset : string
        path to audio dataset
    annotations_file : string, optional
        file name of annotations, by default None
    CustomModel : class, optional
        Custom model to use for the predictions, by default None
    check_if_already_processed : bool, optional
        if you want to force embeddings to be generated again,
        set to True, defaults to True
    min_annotation_length : float, optional
        specify a minimum duration in seconds that has to be exceeded
        for an annotation to be used. This can be useful to filter
        very short annotations, defaults to 0.01 
    overwrite : bool, optional
        If set to False the ground truth will not be recalculated 
        if it already exists. If True, it will produce new ground 
        truth tables based on the annotations with every execution,
        defaults to True

    Returns
    -------
    dict
        dictionary containing report results, ground truth
        array, predictions array, index to label dict and a list
        of the species that weren't found in the classifier
        class list
    """
    model = bacpipe.confirm_model_name(model)
    logger.info("Fetching ground truth and mapping it to model timestamps.\n")
    gt = bacpipe.ground_truth_by_model(
        model,
        audio_dir=dataset,
        annotations_filename=annotations_file,
        bool_filter_labels=False,
        overwrite=overwrite,
        min_annotation_length=min_annotation_length
    )

    # Isolate species columns from metadata
    non_species_labels = [
        "start",
        "end",
        "audiofilename",
        "simultaneous_labels",
    ]
    gt_species_cols = [
        col for col in gt.columns if col not in non_species_labels
    ]
    gt_without_metadata = gt[gt_species_cols].copy()

    loader_obj = bacpipe.run_pipeline_for_single_model(
        model_name=model,
        audio_dir=dataset,
        CustomModel=CustomModel,
        check_if_already_processed=check_if_already_processed,
        **kwargs,
    )

    logger.info("\nFetching model predictions.\n")
    preds, label2idx = loader_obj.predictions(return_type="array")
    if preds is None:
        return {
            "error": "No predictions have been generated, or model does not have classifier."
        }

    label_tuple = associate_ground_truth_and_prediction_labels(
        gt_species_cols, 
        label2idx, 
        gt_without_metadata
    )
    if len(label_tuple) == 1:
        raise AttributeError(
            "No ground truth classes have been found in the predictions."
            "This could be because the model didn't find any of the annotated "
            "species. But it could also be because the model was not trained "
            "to classify the species annotated in the ground truth."
        )
    gt_aligned, shared_labels, shared_indices, not_found = label_tuple

    # 3. Extract and filter matrices in one go
    # Convert predictions to binary (0 or 1) and drop down to shared classes
    gt_binary = gt_aligned[shared_labels].to_numpy()
    pred_binary = (preds[:, shared_indices] > 0).astype(int)

    # 4. Filter out unannotated timestamps
    annotated_mask = gt_binary.sum(axis=1) > 0
    gt_binary = gt_binary[annotated_mask]
    pred_binary = pred_binary[annotated_mask]

    logger.info(f"Annotated timestamps: {annotated_mask.sum()} of {len(preds)}")

    # --- Performance Evaluation ---
    report = classification_report(
        gt_binary,
        pred_binary,
        target_names=shared_labels,
        zero_division=0,
        output_dict=True,
    )

    logger.info("\n--- Overall Report ---")
    logger.info(
        classification_report(
            gt_binary, pred_binary, target_names=shared_labels, zero_division=0
        )
    )
    map_score = average_precision_score(
        gt_binary, 
        preds[:, shared_indices][annotated_mask, :], 
        average="macro"
        )
    logger.info(f"mean Average Precision (mAP): {map_score:.4f}")

    return {
        "report": report,
        "gt_binary": gt_binary,
        "pred_binary": pred_binary,
        "label2idx": label2idx,
        "not_found": not_found,
    }
