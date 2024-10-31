# Copyright (c) Max Planck Institute of Animal Behavior
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utilities
"""
import io
import sys
import matplotlib

matplotlib.use("agg")
import pandas
import torch
import logging
import numpy as np
import collections.abc
import torch.nn as tnn
import matplotlib.pyplot as plt
from itertools import repeat, groupby
# from tensorflow.image import decode_png
from sklearn.metrics import classification_report, average_precision_score
from scipy.ndimage import gaussian_filter1d, gaussian_laplace, minimum_filter1d, maximum_filter1d
# from skimage.filters import threshold_otsu
# from intervaltree import IntervalTree
from itertools import product, chain

logger = logging.getLogger("animal2vec.hydra_train")

from fairseq.logging.meters import Meter
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    TransposeLast,
    # Fp32InstanceNorm
)
from . import SincConv

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message


def rename_attribute(obj, old_name, new_name):
    obj._modules[new_name] = obj._modules.pop(old_name)


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


# n_mels is 96, a 1s segment with sample rate 8kHz
# and hop length of 40 has 200 px
# model = PatchEmbed(patch_size=(96, 2), img_size=(96, 200))
# print(model(torch.zeros(size=(1, 1, 96, 200))).shape)

def get_padding_value(kernel_size):
    kernel_sizes = (kernel_size,)
    from functools import reduce
    from operator import __add__
    conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_sizes[::-1]])
    return conv_padding


def get_conv_size(size, k, p, d, s, dim=2):
    """
    Pass all arguments as a list
    """
    if all([p_ == 0 for p_ in p]):
        if dim == 2:
            p = [np.ceil(s_ / 2).astype(int) for s_ in s]
        else:
            p = [np.ceil(s[0] / 2).astype(int)]
    hh = np.floor((size[0] + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1).astype(int)
    if dim == 2:
        if s[0] == s[1] == 1:
            return size
        ww = np.floor((size[1] + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1).astype(int)
    else:
        if s[0] == 1:
            return size
        ww = None
    return hh, ww


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    pass
    # # Save the plot to a PNG in memory.
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # # Closing the figure prevents it from being displayed directly inside
    # # the notebook.
    # plt.close(figure)
    # buf.seek(0)
    # # Convert PNG buffer to TF image
    # image = decode_png(buf.getvalue(), channels=4)
    # # Add the batch dimension
    # image = torch.tensor(image).unsqueeze(0)
    # return image


def plot_confusion_matrices(confusion_matrix, class_labels=None):
    """
    This build a matplotlib figure of multiple confusion matrices
    """
    num_classes = len(confusion_matrix)
    f, ax = plt.subplots(1, 1, figsize=(1.5 * num_classes, 1.5 * num_classes))
    im = ax.imshow(confusion_matrix, cmap="GnBu", origin="upper")

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.text(-.17, .5, "Truth", rotation=90, ha="center", va="center",
            transform=ax.transAxes, fontsize=30,
            alpha=0.5)
    ax.text(.5, -.1, "Prediction", rotation=0, ha="center", va="center",
            transform=ax.transAxes, fontsize=30,
            alpha=0.5)
    cl = [c[:5] for c in class_labels]
    ax.set_xticklabels(cl, fontsize=20, alpha=0.25)
    ax.set_yticklabels(cl, fontsize=20, alpha=0.25)

    annotate_heatmap(im, valfmt="{x:.0f}", fontsize=16)
    f.tight_layout()
    return plot_to_image(f)


def stack_to_numpy(tensor_list):
    """
    Small helper function that concats a list of torch tensors, detaches it
    from the main graph, moves the array to the cpu and converts it to numpy
    """
    out_tensor = torch.cat(tensor_list)
    return out_tensor.detach().cpu().numpy()


def log_metrics(tensorboard_writer_method, model, stats, cfg, num_updates, train_str="train"):
    with torch.inference_mode():
        unique_labels = eval(cfg.task.unique_labels)
        use_segs = cfg.criterion.segmentation_metrics
        probs = stats["_predictions"].tensor.float().detach().cpu().numpy()  # BTC
        labels = stats["_targets"].tensor.long().detach().cpu().numpy()
        if use_segs:
            source_size = stats["_source_size"].val

        preds = np.where(probs < cfg.criterion.metric_threshold, 0, 1)
        report = pandas.DataFrame(classification_report(labels.reshape(-1, labels.shape[-1]),
                                                        preds.reshape(-1, preds.shape[-1]),
                                                        target_names=unique_labels,
                                                        zero_division=0,
                                                        output_dict=True)).transpose()
        logger.info("\n[Step {:06.0f}] Report on frame-wise metrics with {} as threshold:\n".format(
            num_updates, cfg.criterion.metric_threshold))
        logger.info(report.to_markdown(tablefmt="rounded_outline"))

        if use_segs:
            method_dict = {
                "sigma_s": cfg.criterion.sigma_s,
                "metric_threshold": cfg.criterion.metric_threshold,
                "maxfilt_s": cfg.criterion.maxfilt_s,
                "max_duration_s": cfg.criterion.max_duration_s,
                "lowP": cfg.criterion.lowP,
                "iou_threshold": cfg.criterion.iou_threshold,
            }
            if cfg.criterion.use_focal_loss:
                seg_target_idx = [[get_intervalls(x) for x in y.T] for y in labels]
            else:
                seg_target_idx = [get_intervalls(y) for y in labels]

            # print("\n labels.size()]", probs.shape)
            # for xx in range(1, 8):
            #     print("\n probs[{:02.0f},:,-1]".format(xx), probs[xx, :, -1])
            #     print("\n extracted segments probs", [probs[xx, x:y, -1] for x, y in seg_target_idx[xx][-1]])

            pr, ta, ios, sp, me = model.get_segmented_probs_and_targets(
                {
                    "target": torch.tensor(labels),
                    "seg_target_idx": seg_target_idx,
                    "source_size": source_size
                },
                torch.tensor(probs), method_dict,
                method=cfg.criterion.method
            )
            seg_preds = np.where(pr < cfg.criterion.metric_threshold, 0, 1)
            seg_report = pandas.DataFrame(classification_report(ta, seg_preds,
                                                                target_names=unique_labels,
                                                                zero_division=0,
                                                                output_dict=True)).transpose()
            logger.info("\n[Step {:06.0f}] Report on segmented metrics with {} as threshold:\n".format(
                num_updates, cfg.criterion.metric_threshold))
            logger.info(seg_report.to_markdown(tablefmt="rounded_outline"))

        labels = labels.reshape(-1, labels.shape[-1])
        probs = probs.reshape(-1, probs.shape[-1])
        if callable(tensorboard_writer_method) and cfg.task.with_labels:

            tensorboard_writer = tensorboard_writer_method(train_str)
            tensorboard_writer.add_pr_curve("micro_average",
                                            labels,
                                            probs,
                                            global_step=num_updates)
            map = average_precision_score(labels, probs)
            tensorboard_writer.add_scalar("mAP", map, num_updates)

            if use_segs:
                tensorboard_writer.add_pr_curve("segmented_micro_average",
                                                ta,
                                                pr,
                                                global_step=num_updates)
                seg_map = average_precision_score(ta, pr)
                tensorboard_writer.add_scalar("segmented_mAP", seg_map, num_updates)

            zip_it = [labels.T, probs.T]
            if use_segs:
                zip_it += [ta.T, pr.T, ios.T, sp.T, me.T]
            for class_idx, z in enumerate(zip(*zip_it)):
                tensorboard_writer.add_pr_curve(
                    "class_{}_{}".format(class_idx, unique_labels[class_idx]),
                    z[0], z[1], global_step=num_updates)
                if use_segs:
                    tensorboard_writer.add_pr_curve(
                        "seg_class_{}_{}".format(class_idx, unique_labels[class_idx]),
                        z[2], z[3], global_step=num_updates)

                    to_hist = z[4][z[4] != 0].detach().cpu().numpy().squeeze()  # non-zero IoUs
                    if to_hist.size > 1:  # we need at least two elements
                        # For binning we use sklearns’ default, which is the
                        # maximum of the ‘Sturges’ and ‘Freedman Diaconis’ estimators
                        tensorboard_writer.add_histogram("iou_{}_{}".format(class_idx, unique_labels[class_idx]),
                                                         to_hist,
                                                         global_step=num_updates,
                                                         bins='auto')
                    to_hist = z[5][z[5] != 0].detach().cpu().numpy().squeeze()  # non-zero splits
                    if to_hist.size > 1:  # we need at least two elements
                        # for splits and mergers, we use the nr. of unique vals as number of bins,
                        # as they are integer arrays, where we want a bin for every split and merger that happened.
                        tensorboard_writer.add_histogram("splits_{}_{}".format(class_idx, unique_labels[class_idx]),
                                                         to_hist,
                                                         global_step=num_updates,
                                                         bins=len(np.unique(to_hist)))
                    to_hist = z[6][z[6] != 0].detach().cpu().numpy().squeeze()  # non-zero mergers
                    if to_hist.size > 1:  # we need at least two elements
                        tensorboard_writer.add_histogram("merged_{}_{}".format(class_idx, unique_labels[class_idx]),
                                                         to_hist,
                                                         global_step=num_updates,
                                                         bins=len(np.unique(to_hist)))

            for k, v in report.items():
                if k in ["precision", "recall"]:
                    for kk, vv in v.items():
                        tensorboard_writer.add_scalar("metrics/{}_{}".format(k, kk), vv,
                                                      global_step=num_updates)
            if use_segs:
                for k, v in seg_report.items():
                    if k in ["precision", "recall"]:
                        for kk, vv in v.items():
                            tensorboard_writer.add_scalar("metrics/segmented_{}_{}".format(k, kk), vv,
                                                          global_step=num_updates)
            # mat = net_output["layer_results"]  # Heads x B x T x C
            # mat = mat.clone().detach().sum() / len(mat)  # Heads x B x T x C -> B x T x C
            # one_hot_labels = torch.tensor(labels).reshape([-1, len(unique_labels)]).argmax(-1)
            # metadata = list(np.array(unique_labels)[one_hot_labels.detach().cpu().numpy()])
            # # Reshape to BT x C
            # tensorboard_writer.add_embedding(mat.detach().cpu().numpy().reshape([-1, mat.shape[-1]]),
            #                                  metadata=metadata,
            #                                  global_step=num_updates)
            tensorboard_writer.flush()
        else:
            print(
                "\n[Step {:06.0f}] We don't have a callable tensorboard_writer, so nothing is written out:\n".format(
                    num_updates, cfg.criterion.metric_threshold), flush=True)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def pad_left_right(signal, time, right_pad=False, length_given=False):
    if length_given:
        time_len = time
    else:
        time_len = time.size(0)

    if signal.size(0) == time_len:
        return signal
    if right_pad:
        size_diff = time_len - signal.size(0)
        padded_signal = torch.nn.functional.pad(signal, (0, size_diff), "constant", 0)[:len(time)]
    else:
        size_diff = np.ceil((time_len - signal.size(0)) / 2).astype(int)
        padded_signal = torch.nn.functional.pad(signal, (size_diff, size_diff), "constant", 0)[:len(time)]
    return padded_signal


def get_intervalls(data, shift=0):
    # Group the array in segments with itertools groupby function
    grouped = (list(g) for _, g in groupby(enumerate(data), lambda t: t[1]))
    # Only add the interval if it is with values larger than 0
    return [(g[0][0] + shift, min([len(data) - 1, g[-1][0] + shift])) for g in grouped if g[0][1] == 1]


def avg_pooling(time, preds, method_dict):
    # Do the average pooling
    kernel_size = round(method_dict["fe_sample_rate"] * method_dict["sigma_s"])
    avg_pool_op = torch.nn.AvgPool1d(
        kernel_size=kernel_size,
        stride=1
    )
    avg_pooled_preds = avg_pool_op(preds.view(1, 1, -1))
    # Pad right with zeros to preserve array shape
    windowed_regions = pad_left_right(avg_pooled_preds.view(-1), time, right_pad=True)
    # Threshold to step function
    flattened_regions = torch.where(windowed_regions < method_dict["metric_threshold"], 0., 1.)
    # return intervals of ones
    return get_intervalls(flattened_regions, round(kernel_size / 2))


def max_pooling(time, preds, method_dict):
    # Do the average pooling
    kernel_size = round(method_dict["fe_sample_rate"] * method_dict["sigma_s"])
    max_pool_op = torch.nn.MaxPool1d(
        kernel_size=kernel_size,
        stride=1
    )
    max_pooled_preds = max_pool_op(preds.view(1, 1, -1))
    # Pad right with zeros to preserve array shape
    windowed_regions = pad_left_right(max_pooled_preds.view(-1), time, right_pad=True)
    # Threshold to step function
    flattened_regions = torch.where(windowed_regions < method_dict["metric_threshold"], 0., 1.)
    # return intervals of ones
    return get_intervalls(flattened_regions, round(kernel_size / 2))


def fuse_to_segmented_predictions(time, preds, method_dict, method="avg"):
    # time = time.clone().detach().numpy() / method_dict["sample_rate"]
    assert "sample_rate" in method_dict
    assert "sigma_s" in method_dict
    if method == "max":
        assert "metric_threshold" in method_dict
        assert "fe_sample_rate" in method_dict

        idx_intervals = max_pooling(time, preds, method_dict)
    elif method == "avg":
        assert "metric_threshold" in method_dict
        assert "fe_sample_rate" in method_dict

        idx_intervals = avg_pooling(time, preds, method_dict)
    elif method == "canny":
        assert "maxfilt_s" in method_dict
        assert "max_duration_s" in method_dict
        assert "lowP" in method_dict
        idx_intervals = canny(
            time.clone().detach().cpu().numpy(),
            preds.clone().detach().cpu().numpy(),
            method_dict
        )
    else:
        raise NotImplementedError("{} method not implemented".format(method))
    # Generate a list that holds the absolute time in seconds
    time_intervals = []
    for a in idx_intervals:
        if a[1] != len(time) - 1:
            time_intervals.append([time[a[0]], time[a[1] + 1]])
        else:
            time_intervals.append([time[a[0]], time[-1]])
    # Return time and index intervals
    return time_intervals, idx_intervals


def filter_indicators(signal, indicator, select=np.argmax, debug=False):
    """
    Given an indicator function, find sequences where the indicator is True
    and pick an index based on the values of the signal in the True region.
    Example:
        0 1 2 3 4 5 6 7 8 9  index
        F F T T T F F T T F  indicator
        3 2 4 5 3 2 2 8 7 1 signal
        indicator is true at indices 2,3,4 and then at 7, 8.
        By default, we would return 3, 7:
           index 3 has a maximal value 5 in 4, 5, 3
           index 7 has a maximal value 8 in 8, 7
    :param signal: numpy real valued array
    :param indicator: numpy boolean array
    :param select: selection function (default np.argmax)
    :param debug:  Show list as created
    :return:  list
    """
    ind_list = []
    last = -1
    locations = np.where(indicator)[0]
    start = None
    for idx in locations:
        if debug:
            print(f'{idx} last {last}, {idx - last} : {idx - last > 1}')
        if idx - last > 1:
            if start is not None:
                stat_idx = select(signal[start:idx + 1])
                ind_list.append(start + stat_idx)
            current = start = idx
        else:
            current = idx

        last = idx

    # Handle last one
    if start is not None:
        stat_idx = select(signal[start:locations[-1]])
        ind_list.append(start + stat_idx)

    return ind_list


def canny(time_s, signal, method_dict):
    """
    Canny edge detector
    :param time_s:  time (s) of each sample in signal
    :param signal:  input signal
    :param sigma_s:  size of Gaussian (std dev) in seconds
    :param maxfilt_s:  # time to smooth with maxixmum filter
    :param max_duration_s:  # detections are never longer than max duration (s)
    :param lowP: Low threshold.  Detections are based on Canny edge detector,
      but the detector sometimes misses the minima on some transients if they
      are not sharp enough.  lowP is used for pruning detections.  Detections
      whose Gaussian-smoothed signal is beneath lowP are discarded.
    :param plot:  Plots to aid in interpretation
    :return:  regions of signal as Nx2 matrix of indices
    """
    sigma_s = method_dict["sigma_s"],
    maxfilt_s = method_dict["maxfilt_s"],
    max_duration_s = method_dict["max_duration_s"],
    lowP = method_dict["lowP"]
    N = signal.shape[0]
    adv_s = time_s[1] - time_s[0]
    max_duration_N = max_duration_s / adv_s

    # Smooth the input signal with a Gaussian smooth
    sigma_N = sigma_s // adv_s
    smoothed = gaussian_filter1d(signal, sigma=sigma_N)
    # Use derivative filter to find transitions
    derivative = gaussian_laplace(smoothed, sigma=sigma_N)
    mag = np.abs(derivative)

    # Set the threshold to minimize the intra-class variance on a 2 class
    # problem (Otsu's method)
    tau = 0#threshold_otsu(mag) BUG: threshold_otsu is not defined

    # indicator functions for where we exceed threshold
    # We store filter outputs that we might need again
    maxfilt_N = int(maxfilt_s // adv_s)
    filtered_min_deriv = minimum_filter1d(derivative, maxfilt_N)
    # Strong increases in signal with strong negative derivatives
    maxI = filtered_min_deriv < -tau
    filtered_max_deriv = maximum_filter1d(derivative, maxfilt_N)
    # Strong decreases in signal with strong positive derivatives
    minI = filtered_max_deriv > tau

    # Pick extrema amongst consecutive frames that exceed +/- tau
    min_list = filter_indicators(derivative, minI, select=np.argmax)
    max_list = filter_indicators(derivative, maxI, select=np.argmin)

    # Add boundaries to the min list so that we don't go off the array
    min_list.insert(0, 0)
    min_list.append(N - 1)
    min_values = np.array(min_list)
    max_values = np.array(max_list)
    # Ensure maxima extrema meet criteria, discard weak peaks
    if len(max_values) > 0:
        max_values = max_values[smoothed[max_values] >= lowP]

    # Find the points in min_list that are on either side of the max_list
    # Sometimes, these will be too far away
    left = np.searchsorted(min_values, max_values, side='left') - 1
    right = np.searchsorted(min_values, max_values, side='right')
    intervals = np.vstack((min_values[left], min_values[right])).T

    # Some intervals will be duplicates due to multiple local maxima
    # Sometimes, this is due a missed minima and can result in a very long
    # detection.  We will truncate long detections to twice the maximum
    # duration and then let the long detection truncation routine
    # restrict the duration
    durations = np.diff(intervals, axis=1)
    if np.any(durations <= 0):
        print('<=0 len assertions')
    maxN = max_values.size
    removeI = np.zeros((maxN,), dtype=np.bool_)
    idx = 1
    while idx < maxN:
        first = idx - 1
        last = None
        # Identify all duplicate rows
        while idx < maxN and np.all(intervals[first, :] == intervals[idx, :]):
            last = idx
            idx += 1
        if last is not None:
            # Peaks (max_values) between first:last resulted in duplicate ranges
            # due to undetected minima.  Merge peaks that are close to one
            # another and split the interval on ones that are farther apart
            peaks = max_values[first:last + 1]
            spacing = np.diff(peaks)
            for current in range(len(spacing)):
                target = first + current
                if spacing[current] > max_duration_N:
                    # Too long, probably missed a valley.  Start from peaks
                    # and work towards center.  Truncate when we hit the midpoint
                    # or the signal falls below lowP.
                    midpt = (max_values[target] + max_values[target + 1]) // 2

                    current = max_values[target]
                    while current <= midpt and smoothed[current] > lowP:
                        current += 1
                    intervals[target, 1] = current

                    current = max_values[target + 1]
                    while current > midpt and smoothed[current] > lowP:
                        current -= 1
                    intervals[target + 1, 0] = current
                else:
                    # Remove smaller peak
                    # next_pk is True means second peak is larger
                    next_pk = smoothed[max_values[target + 1]] > \
                              smoothed[max_values[target]]
                    # smaller treated as 0/1 offset
                    removeI[target + next_pk] = True
        idx += 1

    # Remove minor peaks from max_values and intervals
    max_values = max_values[~removeI]
    intervals = intervals[~removeI, :]

    # Repair the very long detections
    # Search on either side until the probability falls beneath lowP
    # or we hit the next detection
    durations = np.diff(intervals, axis=1)
    if np.any(durations <= 0):
        print('<=0 len assertions')

    too_long = np.where(durations * adv_s > max_duration_s)[0]
    if len(too_long):
        # maximum of derivative too small to detect signal, look for low
        # threshold in time domain signal
        for idx in too_long:
            left_idx = current = max_values[idx]
            # latest detection (don't go earlier than this)
            previous = 0 if idx == 0 else np.max(intervals[0:idx, 1])
            while current >= previous and signal[current] >= lowP:
                left_idx = current
                current = current - 1
            right_idx = current = max_values[idx]
            next = N if idx + 1 == intervals.shape[0] else np.min(intervals[idx + 1, 0])
            while current < next and signal[current] >= lowP:
                right_idx = current
                current = current + 1
            max_frames = int(max_duration_s // adv_s)
            if (right_idx - left_idx) > max_frames:
                left_idx = max(left_idx, max_values[idx] - max_frames // 2)
                right_idx = min(right_idx, max_values[idx] + max_frames // 2)
            intervals[idx, :] = [left_idx, right_idx]

    if np.any(durations <= 0):
        print('<=0 len assertions')

    return intervals


def all_equal(iterable):
    if len(iterable) == 0:
        return False
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


class FusedSegmentationMixin:
    """
    Implementing some shared methods using a mixin, as we don't have access to the fairseq parents class
    """

    def __init__(self, *args, **kwargs):
        pass

    def get_iou(self, x, o):
        both_seq_len = sum(a.length() for a in (x, o))
        overlap = x.overlap_size(o)
        if both_seq_len != overlap:
            return overlap / (both_seq_len - overlap)
        else:
            return 1.

    def get_segmented_probs_and_targets(self, sample, probs, method_dict,
                                        method="avg", focal_only=False):
        targets = sample["target"]
        if False:#targets.dim() == probs.dim(): BUG IntervalTree is not defined
            # The indices where an event happens in the target
            fused_targets = sample["seg_target_idx"]
            # Estimate these boundaries using the fuse_predict routine
            _, fused_preds, _ = self.fuse_predict(
                sample_size=sample["source_size"],
                probs=probs,
                method_dict=method_dict,
                method=method
            )
            input_size = targets.size()
            # print(input_size)
            bs = input_size[0]
            time = input_size[1]
            num_classes = input_size[2]

            # The array sizes of 'seg_target' and 'seg_probs' are only reached if there is no overlap
            # between predictions and targets and every segment would have len of 1 with one frame
            # separation between them
            # Pred:     1 0 1 0 1 0 1 0
            # Target:   0 1 0 1 0 1 0 1
            # Then, the segmented prediction intervals are    [(0,1),(2,3),(4,5),(6,7)]
            # and targets                                     [(1,2),(3,4),(5,6),(7,8)]
            # Only then we would end up with a segmented prediction array with the same length
            # of the framewise one -> 8 in this example.

            # Fill with 0
            seg_target = torch.full(input_size,
                                    fill_value=0, dtype=torch.long,
                                    device=probs.device)
            seg_probs = torch.full(input_size,
                                   fill_value=0, dtype=torch.float,
                                   device=probs.device)

            # The max array size for 'seg_overlap_sizes' is half of the time axis
            # This is only reached if there is perfect overlap between preds and
            # targets and every segment would have len of 1 with one frame
            # separation between them
            # Pred:     1 0 1 0 1 0 1 0
            # Target:   1 0 1 0 1 0 1 0
            # Then, the segmented prediction intervals are    [(0,1),(2,3),(4,5),(6,7)]
            # and targets                                     [(0,1),(2,3),(4,5),(6,7)]
            # Only then we would end up with a 'seg_overlap_sizes' array with half the length
            # of the framewise one -> 4 in this example.
            seg_iou = torch.full((bs, round(time / 2), num_classes),
                                 fill_value=0, dtype=torch.float,
                                 device=probs.device)

            # We also track the number of times where a ground truth segment was predicted as
            # two, or more, predicted segments (splits), or when a predicted segment covered
            # multiple ground truth segments (mergers)
            # The max array size for 'seg_overlap_sizes' is floor(len(time) / 3) * 2,
            # This is only reached if there are only 3-frame segments that are consistently
            # split by the predictions / targets, like:
            # Pred:     1 1 1 0 1 1 1 0
            # Target:   1 0 1 0 1 0 1 0
            seg_splits = torch.full((bs, int(np.floor(time / 3) * 2), num_classes),
                                    fill_value=0, dtype=torch.long,
                                    device=probs.device)
            seg_mergers = torch.full((bs, int(np.floor(time / 3) * 2), num_classes),
                                     fill_value=0, dtype=torch.long,
                                     device=probs.device)

            # Iterate across all classes in all samples
            for b, c in product(range(bs), range(num_classes)):
                si = -1  # the counter the the seg probs and targets
                bi = -1  # the counter for the IoUs
                soi = -1  # the counter for the mergers
                smi = -1  # the counter for the mergers

                # we use interval trees for looking for overlaps
                # check if there are "single frame segments" like (25, 25),
                # this would throw an error with IntervalTree
                fu_ta = []
                for t in fused_targets[b][c]:
                    t_ = t
                    if all_equal(t):
                        t_ = (t[0], t[1] + 1)
                    fu_ta.append(t_)
                fused_targets[b][c] = fu_ta

                fu_pr = []
                # print(len(fused_preds[b]))
                for p in fused_preds[b][c]:
                    p_ = p
                    if all_equal(p):
                        p_ = (p[0], p[1] + 1)
                    fu_pr.append(p_)
                fused_preds[b][c] = fu_pr
                ground_truth = IntervalTree.from_tuples(fused_targets[b][c])
                predictions = IntervalTree.from_tuples([[x - (b * time) for x in tu] for tu in fused_preds[b][c]])
                for x_gt in ground_truth:  # iterate through the ground truth
                    ol_pr = predictions.overlap(x_gt)
                    len_ol = len(ol_pr)
                    # current pred has overlap(s) with truth, if len(ol) is larger than 1, than we have splits
                    if len_ol > 0:
                        valid_overlaps = 0
                        for i, o in enumerate(ol_pr):
                            bi += 1
                            si += 1
                            # we use IoU as metric
                            seg_iou[b, bi, c] = self.get_iou(x_gt, o)

                            # check if we have a high enough IoU to be counted as prediction
                            positive_pred = seg_iou[b, bi, c] > method_dict["iou_threshold"]

                            # if IoU is high enough and highest_likelihood is true
                            if positive_pred:
                                valid_overlaps += 1  # true positive(s)
                                seg_target[b, si, c] = 1
                                seg_probs[b, si, c] = probs[b, o[0]: o[1], c].mean()
                            else:  # false positive(s), as IoU was not enough
                                seg_target[b, si, c] = 0
                                seg_probs[b, si, c] = probs[b, x_gt[0]: x_gt[1], c].mean()
                        if valid_overlaps > 1:  # we have a valid split
                            soi += 1
                            seg_splits[b, soi, c] += valid_overlaps  # multiple preds for a single truth -> splits

                    else:  # current pred has no overlap with truth -> this is a false negative
                        si += 1
                        seg_target[b, si, c] = 1
                        seg_probs[b, si, c] = probs[b, x_gt[0]: x_gt[1], c].mean()

                for x_pr in predictions:  # iterate through all predictions
                    ol_gt = ground_truth.overlap(x_pr)
                    len_ol = len(ol_gt)
                    if len_ol > 0:
                        valid_overlaps = 0
                        for i, o in enumerate(ol_gt):
                            # we use IoU as metric
                            if self.get_iou(x_pr, o) > method_dict["iou_threshold"]:
                                valid_overlaps += 1
                        if valid_overlaps > 1:  # we have valid mergers
                            smi += 1
                            seg_mergers[b, smi, c] += valid_overlaps  # multiple truths for a single preds -> merge

                    if len_ol == 0:  # truth has no overlap with pred -> this is a false positive
                        si += 1
                        seg_target[b, si, c] = 0
                        seg_probs[b, si, c] = probs[b, x_pr[0]: x_pr[1], c].mean()

            return (self.prepare_shapes(x, False) for x in (seg_probs, seg_target, seg_iou, seg_splits, seg_mergers))
        else:
            raise NotImplementedError("'get_segmented_logits_and_targets' is currently"
                                      "only implemented for non-one-hot targets")

    def fuse_predict(self, sample_size, probs, method_dict, method="avg", multiplier=0, bs=None):
        # Build the time vector and reshape to account for a single file that is
        # sharded into a single batch
        # the base time is the overal length of a single segment (Default is 10s) times the number in the batch
        # as for fuse_predict a single input file is batched into various smaller files
        base_time = probs.size(0) * sample_size / self.cfg.sample_rate
        # since one batch of len n may not be enough to do inference for the - possibly very long - file
        # we introduce a multiplier parameter, with which the time and idx axis can be shifted.
        # This way the returned time_intervals will always have the correct time stamp and not only relative
        # to the current batch.
        # The bs parameter is to account for the fact than we cannot know the sizes of the previous
        # batches and if the last batch is smaller then the previous ones, then base_time is wrong
        # This correction factor is only used for the inference script.
        if bs is not None:
            start_time = bs * sample_size / self.cfg.sample_rate * (0 + multiplier)
        else:
            start_time = base_time * (0 + multiplier)
        end_time = base_time + start_time
        time_vector_length = probs.size(0) * probs.size(1)
        time = torch.linspace(start_time, end_time, time_vector_length).view(probs.size(0), -1)

        start_index = time_vector_length * (0 + multiplier)
        end_index = time_vector_length * (1 + multiplier)
        indexes = torch.arange(start_index, end_index).view(probs.size(0), -1)
        # Update the dictionary for the fusing routine
        fe_sample_rate = probs.size(1) / sample_size * self.cfg.sample_rate
        method_dict.update({"fe_sample_rate": fe_sample_rate})  # Sampling rate of the feature extractor
        method_dict.update({"sample_rate": self.cfg.sample_rate})  # Sampling rate of the original data

        # This is the kernel size for all methods, if a sample is
        # smaller than that, we continue
        kernel_size = round(method_dict["fe_sample_rate"] * method_dict["sigma_s"])

        # Get the fused intervals
        time_intervals, idx_intervals, likelihoods = [], [], []
        for bp, bt, ind in zip(probs, time, indexes):  # iterate along the batch axis
            if len(bt) < kernel_size:
                continue
            bti, bii, bli = [], [], []
            if bp.dim() == 2:
                for class_probs in bp.t():
                    ti, ii = fuse_to_segmented_predictions(
                        bt, class_probs, method_dict, method=method)
                    bti.append(ti)
                    # print("\n ii", ii)
                    # print("\n ind", ind)
                    # print("\n ind", ind)
                    # print("\n len(ind)", len(ind))
                    bii.append([[ind[x] for x in y] for y in ii])
                    bli.append([class_probs[y[0]:y[1]].mean() for y in ii])
            elif bp.dim() == 1:
                ti, ii = fuse_to_segmented_predictions(
                    bt, bp, method_dict, method=method)  # time intervals is in seconds
                bti.append(ti)
                bii.append([[ind[x] for x in y] for y in ii])
                bli.append([bp[y[0]:y[1]].mean() for y in ii])
            else:
                raise NotImplementedError("Input dimensionality of {} is not supported".format(probs.dim()))
            time_intervals.append(bti)
            idx_intervals.append(bii)
            likelihoods.append(bli)
        # Return time and index intervals
        return time_intervals, idx_intervals, likelihoods


def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    def conf_single_dim(pred, tru):
        confusion_vector = pred / tru
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)

        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives = torch.sum(confusion_vector == 0).item()

        return true_positives, false_positives, true_negatives, false_negatives

    # did we pass multi class or not
    tp, fp, tn, fn = 0, 0, 0, 0,
    if prediction.dim() == 2 and truth.dim() == 2:
        for pc, tc in zip(prediction.t(), truth.t()):  # iterate through the classes
            conf = conf_single_dim(pc, tc)
            tp += conf[0]  # we add everything up for all classes -> micro average
            fp += conf[1]
            tn += conf[2]
            fn += conf[3]

    if prediction.dim() == 1 and truth.dim() == 1:
        conf = conf_single_dim(prediction, truth)
        tp += conf[0]  # we add everything up
        fp += conf[1]
        tn += conf[2]
        fn += conf[3]

    return tp, fp, tn, fn


def sigmoid_focal_loss(inputs: torch.Tensor,
                       targets: torch.Tensor,
                       alpha: float = 0.25,
                       gamma: float = 2,
                       reduction: str = "none") -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape with logits
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = tnn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class ConcatTensorMeter(Meter):
    """Concatenates tensors"""

    def __init__(self, dim=0):
        super().__init__()
        self.reset()
        self.dim = dim

    def reset(self):
        self.tensor = None

    def update(self, val):
        if self.tensor is None:
            self.tensor = val
        else:
            self.tensor = torch.cat([self.tensor, val], dim=self.dim)

    def state_dict(self):
        return {
            "tensor": self.tensor,
        }

    def load_state_dict(self, state_dict):
        self.tensor = state_dict["tensor"]

    @property
    def smoothed_value(self) -> float:
        return 0.  # return a dummy value


class ConvFeatureExtractionModel(tnn.Module):
    def __init__(
            self,
            conv_layers,
            dropout: float = 0.0,
            mode: str = "default",
            conv_bias: bool = False,
            sinc_input: bool = False,
            apply_window_to_root: bool = False,
            sample_rate=8000,
            sinc_norm="layer_norm",
            use_pswish=False
    ):
        super().__init__()
        self.apply_window_to_root = apply_window_to_root
        assert mode in {"default", "layer_norm"}

        def block(
                n_in,
                n_out,
                k,
                stride,
                is_layer_norm=False,
                is_group_norm=False,
                conv_bias=False,
                sinc_input=False,
                apply_window_to_root=False,
                sinc_norm="layer_norm",
                use_pswish=False
        ):
            def make_conv():
                if sinc_input or apply_window_to_root:
                    conv = SincConv(
                        out_channels=n_out,
                        kernel_size=k,
                        stride=stride,
                        sample_rate=sample_rate,
                        learnable_filters=apply_window_to_root and sinc_input,
                        apply_window_to_root=apply_window_to_root,
                        return_abs=True if (sinc_norm == "pcen" or sinc_norm == "instance") else False
                    )
                else:
                    conv = tnn.Conv1d(
                        n_in, n_out, k,
                        stride=stride,
                        bias=conv_bias,
                        padding="same" if stride == 1 else np.ceil(stride / 2).astype(int)
                    )
                    tnn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                           is_layer_norm and is_group_norm
                   ) == False, "layer norm and group norm are exclusive"

            if sinc_norm == "pcen":
                _norm_ = Fp32PCEN(dim)
            elif sinc_norm == "instance":
                _norm_ = Fp32InstanceNorm(dim, affine=True,
                                          track_running_stats=False,
                                          transpose_last=True)
            else:
                _norm_ = Fp32LayerNorm(dim, elementwise_affine=True)

            if is_layer_norm:
                return tnn.Sequential(
                    make_conv(),
                    tnn.Dropout(p=dropout),
                    tnn.Sequential(
                        TransposeLast(),
                        _norm_,
                        TransposeLast(),
                    ),
                    PSwish(num_features=n_out) if ((sinc_input or apply_window_to_root) and use_pswish) else tnn.GELU(),
                )
            elif is_group_norm:
                return tnn.Sequential(
                    make_conv(),
                    tnn.Dropout(p=dropout),
                    _norm_,
                    tnn.GELU(),
                )
            else:
                return tnn.Sequential(
                    make_conv(),
                    tnn.Dropout(p=dropout),
                    tnn.GELU()
                )

        in_d = 1
        self.conv_layers = tnn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                    sinc_input=sinc_input and i == 0,
                    apply_window_to_root=apply_window_to_root and i == 0,
                    sinc_norm=sinc_norm if i == 0 else "layer_norm",
                    use_pswish=use_pswish
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for ii, conv in enumerate(self.conv_layers):
            x = conv(x)

        return x


class PrintLayer(tnn.Module):
    def __init__(self, msg):
        super(PrintLayer, self).__init__()
        self.msg = msg

    def forward(self, x):
        # Do your print / debug stuff here
        print(self.msg, x.size())
        return x


class PCEN(torch.nn.Module):
    """
    This class implements a learnable Per-channel energy normalization (PCEN) layer, supporting both
    original PCEN as specified in [1] as well as sPCEN as specified in [2]

    [1] Yuxuan Wang, Pascal Getreuer, Thad Hughes, Richard F. Lyon, Rif A. Saurous, "Trainable Frontend For
    Robust and Far-Field Keyword Spotting", in Proc of ICASSP 2017 (https://arxiv.org/abs/1607.05666)

    [2] Neil Zeghidour, Olivier Teboul, F{\'e}lix de Chaumont Quitry & Marco Tagliasacchi, "LEAF: A LEARNABLE FRONTEND
    FOR AUDIO CLASSIFICATION", in Proc of ICLR 2021 (https://arxiv.org/abs/2101.08596)

    The default argument values correspond with those used by [2].

    Taken from Spreechbrain
    https://github.com/speechbrain/speechbrain/blob/main/speechbrain/nnet/normalization.py

    We only adapted the routine to work with fp16 and fp32 tensors

    Arguments
    ---------
    input_size : int
        The expected size of the input.
    alpha: float
        specifies alpha coefficient for PCEN
    smooth_coef: float
        specified smooth coefficient for PCEN
    delta: float
        specifies delta coefficient for PCEN
    root: float
        specifies root coefficient for PCEN
    floor: float
        specifies floor coefficient for PCEN
    trainable: bool
        whether to learn the PCEN parameters or use fixed
    per_channel_smooth_coef: bool
        whether to learn independent smooth coefficients for every channel.
        when True, essentially using sPCEN from [2]
    skip_transpose : bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 50, 40])
    >>> pcen = PCEN(40, alpha=0.96)         # sPCEN
    >>> out_tensor = pcen(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 50, 40])
    """

    def __init__(
            self,
            input_size,
            alpha: float = 0.96,
            smooth_coef: float = 0.04,
            delta: float = 2.0,
            root: float = 2.0,
            floor: float = 1e-12,
            trainable: bool = True,
            per_channel_smooth_coef: bool = True,
            skip_transpose: bool = False,
    ):
        super(PCEN, self).__init__()
        self._smooth_coef = smooth_coef
        self._floor = floor
        self._per_channel_smooth_coef = per_channel_smooth_coef
        self.skip_transpose = skip_transpose
        self.alpha = torch.nn.Parameter(
            torch.ones(input_size, dtype=torch.float32) * alpha,
            requires_grad=trainable,
        )
        self.delta = torch.nn.Parameter(
            torch.ones(input_size, dtype=torch.float32) * delta,
            requires_grad=trainable,
        )
        self.root = torch.nn.Parameter(
            torch.ones(input_size, dtype=torch.float32) * root,
            requires_grad=trainable,
        )

        self.ema = ExponentialMovingAverage(
            input_size,
            coeff_init=self._smooth_coef,
            per_channel=self._per_channel_smooth_coef,
            skip_transpose=True,
            trainable=trainable,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channels)
            input to normalize.
        """
        if not self.skip_transpose:
            x = x.transpose(1, -1)
        alpha = torch.min(self.alpha.to(dtype=x.dtype, device=x.device),
                          torch.tensor(1.0, dtype=x.dtype, device=x.device))
        root = torch.max(self.root.to(dtype=x.dtype, device=x.device),
                         torch.tensor(1.0, dtype=x.dtype, device=x.device))
        ema_smoother = self.ema(x)
        one_over_root = 1.0 / root
        output = (
                         x / (self._floor + ema_smoother) ** alpha.view(1, -1, 1)
                         + self.delta.view(1, -1, 1)
                 ) ** one_over_root.view(1, -1, 1) - self.delta.view(
            1, -1, 1
        ) ** one_over_root.view(
            1, -1, 1
        )
        if not self.skip_transpose:
            output = output.transpose(1, -1)
        return output


class ExponentialMovingAverage(torch.nn.Module):
    """
    Applies learnable exponential moving average, as required by learnable PCEN layer

    Arguments
    ---------
    input_size : int
        The expected size of the input.
    coeff_init: float
        Initial smoothing coefficient value
    per_channel: bool
        Controls whether every smoothing coefficients are learned
        independently for every input channel
    trainable: bool
        whether to learn the PCEN parameters or use fixed
    skip_transpose : bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.

    Taken from Spreechbrain
    https://github.com/speechbrain/speechbrain/blob/main/speechbrain/nnet/normalization.py

    We only adapted the routine to work with fp16 and fp32 tensors

    Example
    -------
    >>> inp_tensor = torch.rand([10, 50, 40])
    >>> pcen = ExponentialMovingAverage(40)
    >>> out_tensor = pcen(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 50, 40])
    """

    def __init__(
            self,
            input_size: int,
            coeff_init: float = 0.04,
            per_channel: bool = False,
            trainable: bool = True,
            skip_transpose: bool = False,
    ):
        super(ExponentialMovingAverage, self).__init__()
        self._coeff_init = coeff_init
        self._per_channel = per_channel
        self.skip_transpose = skip_transpose
        self.trainable = trainable
        weights = (
            torch.ones(input_size, dtype=torch.float32) if self._per_channel else torch.ones(1, dtype=torch.float32)
        )
        self._weights = torch.nn.Parameter(
            weights * self._coeff_init, requires_grad=trainable
        )

    def forward(self, x):
        """Returns the normalized input tensor.

       Arguments
        ---------
        x : torch.Tensor (batch, time, channels)
            input to normalize.
        """
        if not self.skip_transpose:
            x = x.transpose(1, -1)
        w = torch.clamp(self._weights, min=0.0, max=1.0)
        initial_state = x[:, :, 0]

        def scan(init_state, x, w):
            """Loops and accumulates."""
            x = x.permute(2, 0, 1)
            acc = init_state
            results = []
            for ix in range(x.shape[0]):
                acc = (w * x[ix]) + ((1.0 - w) * acc)
                results.append(acc.unsqueeze(0))
            results = torch.cat(results, dim=0)
            results = results.permute(1, 2, 0)
            return results

        output = scan(initial_state.to(dtype=x.dtype, device=x.device),
                      x,
                      w.to(dtype=x.dtype, device=x.device))
        if not self.skip_transpose:
            output = output.transpose(1, -1)
        return output.to(dtype=x.dtype, device=x.device)


class Fp32PCEN(PCEN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        return super().forward(input.float()).type_as(input)


class Fp32InstanceNorm(torch.nn.InstanceNorm1d):
    def __init__(self, *args, **kwargs):
        self.transpose_last = "transpose_last" in kwargs and kwargs["transpose_last"]
        if "transpose_last" in kwargs:
            del kwargs["transpose_last"]
        super().__init__(*args, **kwargs)

    def forward(self, input):
        if self.transpose_last:
            input = input.transpose(1, 2)
        output = torch.nn.functional.instance_norm(
            input.float(),
            running_mean=self.running_mean.float() if self.running_mean is not None else None,
            running_var=self.running_var.float() if self.running_var is not None else None,
            weight=self.weight.float() if self.weight is not None else None,
            bias=self.bias.float() if self.bias is not None else None,
            use_input_stats=self.training or not self.track_running_stats,
            momentum=self.momentum,
            eps=self.eps,
        )
        if self.transpose_last:
            output = output.transpose(1, 2)
        return output.type_as(input)


class PSwish(tnn.Module):
    def __init__(self, num_features):
        """
        This module implements PSwish: Swish activation (So SiLU with learnable beta)
        with an additional parameter alpha in front of the sigmoid.
        We initialize the beta parameter with zero and alpha with two.
        This way we start with a linear activation, which is beneficial for the Sinc output
        as these things are initialized as bandpass filters.

            num_features: int, the number of input feature dimensions.
        """
        super(PSwish, self).__init__()
        shape = (1, num_features, 1)
        self.p_swish_alpha = tnn.Parameter(torch.Tensor(*shape))
        self.p_swish_beta = tnn.Parameter(torch.Tensor(*shape))
        self.reset_parameters()

    def forward(self, x):
        return x * self.p_swish_alpha * torch.sigmoid(self.p_swish_beta * x)

    def reset_parameters(self):
        tnn.init.constant_(self.p_swish_alpha, val=2.)
        tnn.init.zeros_(self.p_swish_beta)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def chunk_and_normalize(data, segment_length=10, sample_rate=8000, normalize=True, max_batch_size=16):
    """
    This is a helper function that chunks an input array into segment_length long chunks
    and (optionally) normalizes each chunk to zero mean and unit variance.
    """
    data = data.squeeze()
    assert data.ndim == 1
    seq_len = round(segment_length * sample_rate)
    if len(data) > seq_len:
        # we need to split the input file into smaller segments
        batched_wav = list(data.split(seq_len))
        # The last segment will have a different length than the others. We right pad with zero
        batched_wav[-1] = pad_left_right(batched_wav[-1], batched_wav[0], right_pad=True)
        # If the batched wav file is longer then our max batch_size, then chunk it
        if len(batched_wav) > max_batch_size:
            batched_wav = list(chunks(batched_wav, max_batch_size))
        else:
            # place in list such that it is a single batch when passed to model
            batched_wav = [batched_wav]
    else:
        batched_wav = [data]

    if normalize:
        b_ = []
        for batch in batched_wav:
            if not torch.is_tensor(batch):
                batch = torch.stack(batch)  # stack the list of tensors
            elif batch.dim() == 1:  # split segments or single segment
                batch = batch.view(1, -1)
            b_.append([torch.nn.functional.layer_norm(x, x.shape).squeeze() for x in batch])
        batched_wav = b_
    return batched_wav
