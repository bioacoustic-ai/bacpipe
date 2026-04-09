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

    loader_obj = bacpipe.run_pipeline_for_single_model(
        model_name=model,
        audio_dir=dataset,
        CustomModel=CustomModel,
        check_if_already_processed=check_if_already_processed,
        **kwargs
    )
    
    print('\nFetching model predictions.\n')
    preds, label2idx = loader_obj.predictions(return_type='array')
    
    # Align ground truth labels to predicted label indices
    ground_truth_array = gt['label:species']
    not_found = []

    print(
        'The following species were found in the ground truth '
        'and the predictions:'
        )
    for label, idx in gt['label_dict:species'].items():
        if label in label2idx:
            print(label)
            ground_truth_array[gt['label:species'] == idx] = label2idx[label]
        else:
            not_found.append(label)
    
    if not_found:
        print(
            '\nThese species were found in the ground truth but '
            'NOT in the predictions:',
            not_found
            )
        l2i_regex = {re.sub(r'[-\s]', '', label).lower(): i for label, i in label2idx.items()}
        for label in not_found:
            label_regex = re.sub(r'[-\s]', '', label).lower()
            if label_regex in l2i_regex:
                print('With regex we found', label)
                ground_truth_array[gt['label:species'] == idx] = l2i_regex[label_regex]
                not_found.remove(label)
        print('With regex we still did not find:', not_found)
                
                
    # Build binary matrices using l2i as column ordering
    n_timestamps = len(preds)
    n_classes = len(label2idx)

    gt_binary = np.zeros((n_timestamps, n_classes), dtype=int)
    pred_binary = preds
    pred_binary[pred_binary > 0] = 1

    for label, col_idx in label2idx.items():
        gt_binary[np.any(ground_truth_array == col_idx, axis=1), col_idx] = 1

    # Filter to columns that appear in ground truth
    gt_classes = set(ground_truth_array[ground_truth_array > -1].astype(int))
    pred_classes = set(label2idx.values())
    all_classes = gt_classes.intersection(pred_classes)
    
    gt_binary = gt_binary[:, list(all_classes)]
    pred_binary = pred_binary[:, list(all_classes)]

    # Filter out unannotated timestamps
    annotated_mask = gt_binary.sum(axis=1) > 0
    gt_binary = gt_binary[annotated_mask]
    pred_binary = pred_binary[annotated_mask]

    print(f"annotated timestamps: {annotated_mask.sum()} of {n_timestamps}")

    # Evaluate performance
    idx2label = {v: k for k, v in label2idx.items()}
    target_names = [idx2label[i] for i in all_classes]

    report = classification_report(
        gt_binary,
        pred_binary,
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )
    print("\n--- Overall Report ---")
    print(classification_report(
        gt_binary,
        pred_binary,
        target_names=target_names,
        zero_division=0
    ))

    return {
        'report': report,
        'gt_binary': gt_binary,
        'pred_binary': pred_binary,
        'label2idx': label2idx,
        'not_found': not_found
    }

