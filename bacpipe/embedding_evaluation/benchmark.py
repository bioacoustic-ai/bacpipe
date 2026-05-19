import bacpipe
import numpy as np
import re

from sklearn.metrics import classification_report

def benchmark(
    model, dataset, 
    annotations_file=None, 
    CustomModel=None,
    check_if_already_processed=True,
    **kwargs
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

    Returns
    -------
    dict
        dictionary containing report results, ground truth 
        array, predictions array, index to label dict and a list
        of the species that weren't found in the classifier 
        class list
    """
    print('Fetching ground truth and mapping it to model timestamps.\n')
    gt = bacpipe.ground_truth_by_model(
        model,
        audio_dir=dataset,
        annotations_filename=annotations_file,
        single_label=False,
        bool_filter_labels=False,
        overwrite=True
    )
    
    # Isolate species columns from metadata
    non_species_labels = ['starts', 'ends', 'audiofilename', 'species_richness']
    gt_species_cols = [col for col in gt.columns if col not in non_species_labels]
    gt_without_metadata = gt[gt_species_cols]

    loader_obj = bacpipe.run_pipeline_for_single_model(
        model_name=model,
        audio_dir=dataset,
        CustomModel=CustomModel,
        check_if_already_processed=check_if_already_processed,
        **kwargs
    )
    
    print('\nFetching model predictions.\n')
    preds, label2idx = loader_obj.predictions(return_type='array')
    if preds is None:
        return {'error': "No predictions have been generated, or model does not have classifier."}
    
    # --- Class Matching & Validation ---
    def clean_string(s):
        return re.sub(r'[-\s]', '', s).lower()

    # Find exact matching classes
    found = [label for label in gt_species_cols if label in label2idx]
    not_found = [label for label in gt_species_cols if label not in label2idx]

    print('The following species were found in the ground truth and the predictions:')
    for label in found:
        print(f" - {label}")
    
    # Fallback to Regex matching for missing classes
    if not_found:
        print(f'\nSpecies found in ground truth but NOT exactly in predictions: {not_found}')
        
        l2i_regex = {clean_string(lbl): lbl for lbl in label2idx.keys()}
        still_not_found = []
        
        for label in not_found:
            cleaned = clean_string(label)
            if cleaned in l2i_regex:
                matched_label = l2i_regex[cleaned]
                print(f"With regex we matched ground truth '{label}' to prediction '{matched_label}'")
                found.append(label)
            else:
                still_not_found.append(label)
                
        not_found = still_not_found
        print(f'Remaining unmatched species: {not_found}')
                
    if not found:
        return {'error': "No ground truth classes have been found in the predictions."}

    # --- Matrix Generation & Alignment (Condensed) ---
    # 1. Align ground truth to label2idx columns, filling missing ones with 0
    gt_aligned = gt_without_metadata.reindex(columns=label2idx.keys(), fill_value=0)
    
    # 2. Get shared labels and their corresponding integer indices
    shared_labels = [lbl for lbl in label2idx.keys() if lbl in gt_without_metadata.columns]
    shared_indices = [label2idx[lbl] for lbl in shared_labels]
    
    # 3. Extract and filter matrices in one go
    # Convert predictions to binary (0 or 1) and drop down to shared classes
    gt_binary = gt_aligned[shared_labels].to_numpy()
    pred_binary = (preds[:, shared_indices] > 0).astype(int)

    # 4. Filter out unannotated timestamps
    annotated_mask = gt_binary.sum(axis=1) > 0
    gt_binary = gt_binary[annotated_mask]
    pred_binary = pred_binary[annotated_mask]

    print(f"Annotated timestamps: {annotated_mask.sum()} of {len(preds)}")

    # --- Performance Evaluation ---
    report = classification_report(
        gt_binary,
        pred_binary,
        target_names=shared_labels,
        zero_division=0,
        output_dict=True
    )
    
    print("\n--- Overall Report ---")
    print(classification_report(
        gt_binary,
        pred_binary,
        target_names=shared_labels,
        zero_division=0
    ))

    return {
        'report': report,
        'gt_binary': gt_binary,
        'pred_binary': pred_binary,
        'label2idx': label2idx,
        'not_found': not_found
    }