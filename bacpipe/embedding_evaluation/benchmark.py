import bacpipe
import numpy as np
bacpipe.settings.main_results_dir = '/mnt/swap/Work/Embeddings'
import re

from sklearn.metrics import classification_report

def benchmark(model, dataset, annotations_file=None):
    
    gt = bacpipe.ground_truth_by_model(
        model, 
        audio_dir=dataset,
        annotations_filename=annotations_file,
        single_label=False,
        bool_filter_labels=False,
        overwrite=True
        )

    # print_output(gt)

    loader_obj = bacpipe.run_pipeline_for_single_model(
        model_name=model,
        audio_dir=dataset
    )

    # loader_obj.embeddings(as_type='array')
    preds, idx2label = loader_obj.predictions(threshold=0.5, as_type='array')
    l2i = {v: k for k, v in idx2label.items()}
    # benchmark HSN dataset using birdnet
    ground_truth_array = gt['label:species']
    not_found = []
    for label, idx in gt['label_dict:species'].items():
        # if label == 'Eurasian Kestrel':
        #     label = 'Common Kestrel'
        if label in l2i:
            print('found', label)
            ground_truth_array[gt['label:species']==idx] = l2i[label]
        else:
            print('not found', label)
            not_found.append(label)
    
    if not_found:
        l2i_regex = {re.sub(r'[-\s]', '', label).lower(): i for label, i in l2i.items()}
        not_found_dict = {k: v for k, v in gt['label_dict:species'].items() if k in not_found}
        for label, idx in not_found_dict.items():
            label_regex = re.sub(r'[-\s]', '', label).lower()
            if label_regex in l2i_regex:
                print('With regex we found', label)
                ground_truth_array[gt['label:species']==idx] = l2i_regex[label_regex]
            else:
                print('Still not found', label)
                not_found.append(label)
            
            
            
    # --- Convert ragged -1-padded arrays to binary indicator matrices ---
    all_classes = np.unique(ground_truth_array[ground_truth_array > -1])
    n_classes = len(all_classes)
    class_to_col = {cls: i for i, cls in enumerate(all_classes)}
    n_timestamps = len(preds)

    gt_binary = np.zeros((n_timestamps, n_classes), dtype=int)
    pred_binary = np.zeros((n_timestamps, n_classes), dtype=int)

    for i in range(n_timestamps):
        for cls in ground_truth_array[i]:
            if cls > -1 and cls in class_to_col:
                gt_binary[i, class_to_col[cls]] = 1
        for cls in preds[i]:
            if cls > -1 and cls in class_to_col:
                pred_binary[i, class_to_col[cls]] = 1

    # --- Filter out timestamps with no ground truth annotation ---
    annotated_mask = gt_binary.sum(axis=1) > 0
    gt_binary = gt_binary[annotated_mask]
    pred_binary = pred_binary[annotated_mask]

    # --- Metrics ---
    print("\n--- Overall Report ---")
    print(classification_report(
        gt_binary,
        pred_binary,
        target_names=[idx2label[cls] for cls in all_classes],
        zero_division=0
    ))

    return {
        'gt_binary': gt_binary,
        'pred_binary': pred_binary,
        'all_classes': all_classes,
        'idx2label': idx2label,
        'class_to_col': class_to_col
    }




benchmark('birdnet', '/media/siriussound/Extreme SSD/Recordings/terrestrial/Birds/BirdSet/HSN - soundscapes high Sierra Nevada', annotations_file='annotations_longname.csv')







if False:
    
    loader = bacpipe.run_pipeline_for_single_model(
        model, dataset
    )
    preds = loader.predictions()

    # fnc = bacpipe.core.workflows.make_set_paths_func(bacpipe.config.audio_dir, **vars(bacpipe.settings))
    # paths = fnc(model)
            
    from bacpipe.embedding_evaluation.label_embeddings import ground_truth_by_model
    gt = ground_truth_by_model(
        model, 
        audio_dir=dataset,
        annotations_filename=annotations_file,
        single_label=False,
        min_annotation_length=0.05
        )


    # per_file_classifications_folder = paths.preds_path / 'original_classifier_outputs'
    # annots = per_file_classifications_folder.rglob('*.json')
    # import json
    # preds = {}
    # for annot in annots:
    #     with open(annot, 'r') as a:
    #         key = annot.relative_to(paths.preds_path / 'original_classifier_outputs')
    #         preds[key] = json.load(a)
            
    p = list(preds.values())[0]
    pp = p
    pp.pop('head')
    model_length = 3
    import pandas as pd
    import numpy as np
    df = pd.DataFrame()
    start = []
    end = []
    audiofilename = []
    species = []
    for file, values in preds.items():
        for k, v in pp.items():
            if k == 'head':
                continue
            start.extend(np.array(v['time_bins_exceeding_threshold']) * model_length)
            end.extend((np.array(v['time_bins_exceeding_threshold']) + 1) * model_length)
            audiofilename.extend(
                [file] * len(v['time_bins_exceeding_threshold'])
                )
            species.extend([k] * len(v['time_bins_exceeding_threshold']))
    df['start'] = start
    df['end'] = end
    df['audiofilename'] = audiofilename
    df['label:species'] = species
    l2i = {v: k for k, v in enumerate(df['label:species'].unique())}
    meta_dict = {'label:species': l2i}

    pred = ground_truth_by_model(
        model_name, 
        label_df = df,
        label_idx_dict=meta_dict,
        audio_dir=bacpipe.config.audio_dir,
        annotations_filename='annotations.csv',
        single_label=False,
        min_annotation_length=0.05
        )

    from bacpipe.embedding_evaluation.label_embeddings import get_ground_truth
    gt = get_ground_truth(model_name)








    ground_truth_classes = np.unique(ground_truth_array[ground_truth_array>-1])

    tp, fp, tn, fn = [], [], [], []
    for ts, (pr_label, gt_label) in enumerate(zip(preds, ground_truth_array)):
        if set(pr_label) == {-1} and not set(gt_label) == {-1}:
            fn.append(ts)
        elif set(pr_label) == {-1} and set(gt_label) == {-1}:
            tn.append(ts)
        else:
            for pr_class in pr_label:
                if pr_class == -1:
                    continue
                if pr_class in gt_label:
                    tp.append(pr_class)
                elif not pr_class in gt_label:
                    fp.append(pr_class)
    overall_accuracy = len(tp) / len(preds)
    classwise_accuracy = {}
    p_classes, p_counts = np.unique(tp, return_counts=True)
    gt_classes, gt_counts = np.unique(ground_truth_array[ground_truth_array>-1], return_counts=True)
    classwise_groundtrurh = {cls: cnt for cls, cnt in zip(gt_classes, gt_counts)}
    for p_class, p_count in zip(p_classes, p_counts):
        if not p_class in classwise_groundtrurh:
            continue
        gt_cnt = classwise_groundtrurh[p_class]
        classwise_accuracy[p_class] = p_count / gt_cnt