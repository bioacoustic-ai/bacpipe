import bacpipe


def benchmark(model, dataset, annotations_file=None):
    model_name = 'audioprotopnet'

    fnc = bacpipe.core.workflows.make_set_paths_func(bacpipe.config.audio_dir, **vars(bacpipe.settings))
    paths = fnc(model_name)
            
    from bacpipe.embedding_evaluation.label_embeddings import ground_truth_by_model
    gt = ground_truth_by_model(
        model_name, 
        audio_dir=bacpipe.config.audio_dir,
        annotations_filename='annotations.csv',
        single_label=False,
        min_annotation_length=0.05
        )


    per_file_classifications_folder = paths.preds_path / 'original_classifier_outputs'
    annots = per_file_classifications_folder.rglob('*.json')
    import json
    preds = {}
    for annot in annots:
        with open(annot, 'r') as a:
            key = annot.relative_to(paths.preds_path / 'original_classifier_outputs')
            preds[key] = json.load(a)
            
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








    gt = bacpipe.ground_truth_by_model(
        'birdnet', 
        audio_dir='bacpipe/tests/test_data',
        annotations_filename='annotations.csv',
        single_label=False
        )

    # print_output(gt)

    loader_obj = bacpipe.run_pipeline_for_single_model(
        model_name='birdnet',
        audio_dir='bacpipe/tests/test_data'
    )

    # loader_obj.embeddings(as_type='array')
    preds, idx2label = loader_obj.predictions(threshold=0.5, as_type='array')
    l2i = {v: k for k, v in idx2label.items()}
    # benchmark HSN dataset using birdnet
    ground_truth_array = gt['label:species']
    for label, idx in gt['label_dict:species'].items():
        if label == 'Eurasian Kestrel':
            label = 'Common Kestrel'
        ground_truth_array[gt['label:species']==idx] = l2i[label]
    import numpy as np
    ground_truth_classes = np.unique(ground_truth_array[ground_truth_array>-1])

    gl_tp, gl_fp, gl_tn, gl_fn = [[]] * 4
    tp, fp, tn, fn = [], [], [], []
    for ts, (pr, gt) in enumerate(zip(preds, ground_truth_array)):
        if set(pr) == {-1} and not set(gt) == {-1}:
            fn.append(ts)
        elif set(pr) == {-1} and set(gt) == {-1}:
            tn.append(ts)
        else:
            for pr_class in pr:
                if pr_class == -1:
                    continue
                if pr_class in gt:
                    tp.append(pr_class)
                elif not pr_class in gt:
                    fp.append(pr_class)
    overall_accuracy = len(tp) / ts
    classwise_accuracy = {}
    p_classes, p_counts = np.unique(tp, return_counts=True)
    gt_classes, gt_counts = np.unique(ground_truth_array[ground_truth_array>-1], return_counts=True)
    classwise_groundtrurh = {cls: cnt for cls, cnt in zip(gt_classes, gt_counts)}
    for p_class, p_count in zip(p_classes, p_counts):
        if not p_class in classwise_groundtrurh:
            continue
        gt_cnt = classwise_groundtrurh[p_class]
        classwise_accuracy[p_class] = p_count / gt_cnt
