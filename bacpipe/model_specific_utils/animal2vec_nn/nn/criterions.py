# Copyright (c) Max Planck Institute of Animal Behavior
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utilities
"""
import math
import torch
from omegaconf import II
from dataclasses import dataclass, field
from fairseq import utils, metrics
from fairseq.logging.meters import safe_round
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)
from fairseq.criterions.model_criterion import ModelCriterion, ModelCriterionConfig
from . import confusion, sigmoid_focal_loss, ConcatTensorMeter
from sklearn import metrics as sklearn_metrics
import numpy as np

try:
    from torch import _assert
except ImportError:

    def _assert(condition: bool, message: str):
        assert condition, message


@dataclass
class LabelSmoothedCrossEntropyCriterionConfigModifiedLogs(
    LabelSmoothedCrossEntropyCriterionConfig
):
    unique_labels: str = II("task.unique_labels")
    verbose_tensorboard_logging: str = II("task.verbose_tensorboard_logging")
    segmentation_metrics: bool = field(
        default=False,
        metadata={"help": "if set, we use segmented metrics (really slow)"},
    )
    use_focal_loss: bool = field(
        default=True,
        metadata={"help": "if set, we use the focal loss instead of label smoothed CE"},
    )
    metric_threshold: float = field(
        default=0.25,
        metadata={"help": "The minimum likelihood which is deemed a prediction"},
    )
    iou_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The minimum IoU that is needed for a prediction to be counted as "
            "overlapping. 0 means that even a single frame overlap is enough, "
            "while 1 means that only a perfect overlap is counted as overlap."
        },
    )
    sigma_s: float = field(
        default=0.1,
        metadata={
            "help": "Size of Gaussian (std dev) in seconds for the canny method. "
            "Filter width in seconds for avg and max methods."
        },
    )
    maxfilt_s: float = field(
        default=0.1,
        metadata={
            "help": "Time to smooth with maxixmum filter. Only used when method=canny"
        },
    )
    max_duration_s: float = field(
        default=0.5,
        metadata={
            "help": "Detections are never longer than max duration (s). Only used when method=canny"
        },
    )
    lowP: float = field(
        default=0.125,
        metadata={
            "help": "Low threshold. Detections are based on Canny edge detector, "
            "but the detector sometimes misses the minima on some transients"
            "if they are not sharp enough.  lowP is used for pruning detections."
            "Detections whose Gaussian-smoothed signal is beneath lowP are discarded."
            "Only used when method=canny"
        },
    )
    method: str = field(
        default="avg",
        metadata={
            "help": "Which method to use for fusing the predictions into time bins."
            "avg: Average pooling, then thresholding."
            "max: Max pooling, then thresholding."
            "canny: Canny edge detector"
        },
    )

    can_sum: bool = True


@dataclass
class ExpandedModelCriterionConfig(ModelCriterionConfig):
    unique_labels: str = II("task.unique_labels")
    verbose_tensorboard_logging: str = II("task.verbose_tensorboard_logging")
    segmentation_metrics: bool = field(
        default=False,
        metadata={"help": "if set, we use segmented metrics (really slow)"},
    )
    use_focal_loss: bool = field(
        default=True,
        metadata={"help": "if set, we use the focal loss instead of label smoothed CE"},
    )
    metric_threshold: float = field(
        default=0.25,
        metadata={"help": "The minimum likelihood which is deemed a prediction"},
    )
    iou_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The minimum IoU that is needed for a prediction to be counted as "
            "overlapping. 0 means that even a single frame overlap is enough, "
            "while 1 means that only a perfect overlap is counted as overlap."
        },
    )
    sigma_s: float = field(
        default=0.1,
        metadata={
            "help": "Size of Gaussian (std dev) in seconds for the canny method. "
            "Filter width in seconds for avg and max methods."
        },
    )
    maxfilt_s: float = field(
        default=0.1,
        metadata={
            "help": "Time to smooth with maxixmum filter. Only used when method=canny"
        },
    )
    max_duration_s: float = field(
        default=0.5,
        metadata={
            "help": "Detections are never longer than max duration (s). Only used when method=canny"
        },
    )
    lowP: float = field(
        default=0.125,
        metadata={
            "help": "Low threshold. Detections are based on Canny edge detector, "
            "but the detector sometimes misses the minima on some transients"
            "if they are not sharp enough.  lowP is used for pruning detections."
            "Detections whose Gaussian-smoothed signal is beneath lowP are discarded."
            "Only used when method=canny"
        },
    )
    method: str = field(
        default="avg",
        metadata={
            "help": "Which method to use for fusing the predictions into time bins."
            "avg: Average pooling, then thresholding."
            "max: Max pooling, then thresholding."
            "canny: Canny edge detector"
        },
    )

    can_sum: bool = True


@register_criterion(
    "finetunecriterion", dataclass=LabelSmoothedCrossEntropyCriterionConfigModifiedLogs
)
class FinetuneCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    """
    The clones the fairseq LabelSmoothed CE Loss and adds some logging macros and the focal loss
    """

    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        unique_labels,
        segmentation_metrics,
        use_focal_loss,
        metric_threshold,
        iou_threshold,
        sigma_s,
        maxfilt_s,
        max_duration_s,
        lowP,
        method,
        can_sum,
        verbose_tensorboard_logging,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(
            task=task,
            sentence_avg=sentence_avg,
            label_smoothing=label_smoothing,
            ignore_prefix_size=ignore_prefix_size,
            report_accuracy=report_accuracy,
        )
        self.num_classes = len(eval(unique_labels))
        self.segmentation_metrics = segmentation_metrics
        self.use_focal_loss = use_focal_loss
        self.metric_threshold = metric_threshold
        self.iou_threshold = iou_threshold
        self.sigma_s = sigma_s
        self.maxfilt_s = maxfilt_s
        self.max_duration_s = max_duration_s
        self.lowP = lowP
        self.method = method
        self.can_sum = can_sum
        self.can_sum_original = can_sum
        self.verbose_tensorboard_logging = verbose_tensorboard_logging

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            if target.dim() == lprobs.dim():
                target = target[:, self.ignore_prefix_size :, :].contiguous()
            else:
                target = target[:, self.ignore_prefix_size :].contiguous()

        if target.dim() == lprobs.dim():
            target = target.view(-1, target.size(-1))
        else:
            target = target.view(-1)
        return lprobs.view(-1, lprobs.size(-1)), target.to(torch.int64)

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        if target.dim() == lprobs.dim():
            preds = torch.sigmoid(
                model.get_logits(net_output).view(-1, lprobs.size(-1))
            )
            preds = torch.where(preds < self.metric_threshold, 0, 1)
            n_correct = torch.sum(preds.eq(target))
            total = torch.tensor(target.shape).prod()
        else:
            # mask = torch.ones_like(target, dtype=torch.bool)
            mask = target.ne(self.padding_idx)
            # print("\n\n mask", mask)
            # print("\n\n lprobs", lprobs)
            # print("\n\n lprobs.argmax(-1)", lprobs.argmax(-1))
            selected_lprobs = lprobs.argmax(1).masked_select(mask)
            selected_targets = target.masked_select(mask)
            n_correct = torch.sum(selected_lprobs.eq(selected_targets))
            total = torch.sum(mask)

        return n_correct.squeeze(), total.squeeze()

    def compute_prec_rec_f1(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        if target.dim() == lprobs.dim():
            preds = torch.sigmoid(
                model.get_logits(net_output).view(-1, lprobs.size(-1))
            )
        else:
            preds = torch.softmax(model.get_logits(net_output), -1)
            target = torch.nn.functional.one_hot(target.long(), preds.size(-1))
        preds = torch.where(preds < self.metric_threshold, 0, 1)
        tp, fp, tn, fn = confusion(preds, target)

        return tp, fp, tn, fn

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if model.training:
            self.can_sum = self.can_sum_original
        net_output = model(**sample["net_input"])
        if self.use_focal_loss:
            logits = model.get_logits(net_output)
            target = model.get_targets(sample, net_output)
            reduction = "none" if not reduce else "sum"
            loss = sigmoid_focal_loss(logits, target, reduction=reduction)
            nll_loss = torch.tensor(0.0)
        else:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            tp, fp, tn, fn = self.compute_prec_rec_f1(model, net_output, sample)
            logging_output["finetune/n_correct"] = utils.item(n_correct.data)
            logging_output["finetune/total"] = utils.item(total.data)
            logging_output["finetune/tp"] = utils.item(tp)
            logging_output["finetune/fp"] = utils.item(fp)
            logging_output["finetune/tn"] = utils.item(tn)
            logging_output["finetune/fn"] = utils.item(fn)
        if self.verbose_tensorboard_logging and not model.training:
            self.can_sum = False
            logging_output["_predictions"] = torch.sigmoid(
                model.get_logits(net_output, reshape=False)
            )
            logging_output["_targets"] = model.get_targets(
                sample, net_output, reshape=False
            )
            if self.segmentation_metrics:
                logging_output["_source_size"] = sample["net_input"]["source"].size(-1)

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )

        metrics.log_scalar("misc/ntokens", ntokens)
        metrics.log_scalar("misc/nsentences", nsentences)
        metrics.log_scalar("misc/sample_size", sample_size)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "misc/nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "misc/perplexity",
            lambda meters: utils.get_perplexity(meters["misc/nll_loss"].avg),
        )

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar("finetune/total", total)

        builtin_keys = {
            "loss",
            "nll_loss",
            "ntokens",
            "nsentences",
            "sample_size",
            "total",
        }

        for k in logging_outputs[0]:
            # print("Key: {} ...".format(k), end="")
            if k not in builtin_keys and not k.startswith("_"):
                # print("added")
                val = sum(
                    log.get(k, 0) for log in logging_outputs
                )  # sum across devices
                if k.startswith("loss"):
                    metrics.log_scalar(
                        "individual_losses/" + k,
                        val / (sample_size or 1) / math.log(2),
                        sample_size,
                        round=3,
                    )
                else:
                    metrics.log_scalar(k, val / len(logging_outputs), round=3)
            # else:
            #     print("not added")

        if utils.item(sum(log.get("finetune/total", 0) for log in logging_outputs)) > 0:
            metrics.log_derived(
                "metrics/finetune/accuracy",
                lambda meters: (
                    safe_round(
                        meters["finetune/n_correct"].sum
                        * 100.0
                        / meters["finetune/total"].sum,
                        3,
                    )
                    if meters["finetune/total"].sum > 0
                    else float("nan")
                ),
            )
            metrics.log_derived(
                "metrics/finetune/precision",
                lambda meters: (
                    safe_round(
                        meters["finetune/tp"].sum
                        * 100.0
                        / (meters["finetune/tp"].sum + meters["finetune/fp"].sum),
                        3,
                    )
                    if meters["finetune/tp"].sum + meters["finetune/fp"].sum > 0
                    else float("nan")
                ),
            )
            metrics.log_derived(
                "metrics/finetune/recall",
                lambda meters: (
                    safe_round(
                        meters["finetune/tp"].sum
                        * 100.0
                        / (meters["finetune/tp"].sum + meters["finetune/fn"].sum),
                        3,
                    )
                    if meters["finetune/tp"].sum + meters["finetune/fn"].sum > 0
                    else float("nan")
                ),
            )
            metrics.log_derived(
                "metrics/finetune/f1",
                lambda meters: (
                    safe_round(
                        meters["finetune/tp"].sum
                        * 100.0
                        * 2
                        / (
                            2 * meters["finetune/tp"].sum
                            + meters["finetune/fn"].sum
                            + meters["finetune/fp"].sum
                        ),
                        3,
                    )
                    if 2 * meters["finetune/tp"].sum
                    + meters["finetune/fn"].sum
                    + meters["finetune/fp"].sum
                    > 0
                    else float("nan")
                ),
            )

        for kk in ["_predictions", "_targets", "_source_size"]:
            if kk in logging_outputs[0]:
                if kk == "_source_size":
                    metrics.log_scalar(kk, logging_outputs[0].get(kk, 0), round=1)
                else:
                    metrics.log_custom(
                        ConcatTensorMeter,
                        kk,
                        torch.cat([logs[kk].cpu() for logs in logging_outputs], dim=0),
                    )

    def logging_outputs_can_be_summed(self) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return self.can_sum


@register_criterion("expanded_model", dataclass=ExpandedModelCriterionConfig)
class ExpandedModelCriterion(ModelCriterion):
    """
    This criterion relies on the model to supply losses.
    The losses should be a dictionary of name -> scalar returned by
    the model either by including it in the net_output dict or by
    implementing a get_losses(net_output, sample) method. The final loss is
    a scaled sum of all losses according to weights in loss_weights.
    If no weights are provided, then all losses are scaled by 1.0.
    The losses will be automatically logged. Additional keys from
    net_output dict can be logged via the log_keys parameter.
    """

    def __init__(self, task, loss_weights=None, log_keys=None, can_sum=True):
        self.can_sum_original = can_sum
        super().__init__(task, loss_weights, log_keys, can_sum)

    def forward(self, model, sample, reduce=True):
        if model.training:
            self.can_sum = self.can_sum_original
        loss, sample_size, logging_output = super().forward(model, sample, reduce)
        if not model.training:
            self.can_sum = False
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        metrics.log_scalar("sample_size", sample_size)

        builtin_keys = {"loss", "ntokens", "nsentences", "sample_size", "_world_size"}

        world_size = utils.item(
            sum(log.get("_world_size", 0) for log in logging_outputs)
        )

        for k in logging_outputs[0]:
            if k not in builtin_keys and not k.startswith("_"):
                val = sum(log.get(k, 0) for log in logging_outputs)
                if k.startswith("loss_"):
                    metrics.log_scalar(k, val / sample_size, sample_size, round=3)
                elif k.startswith("pretrain/"):
                    metrics.log_scalar(k, val, round=3)
                else:
                    metrics.log_scalar(k, val / world_size, round=3)

        total = sum(log.get("pretrain/total", 0) for log in logging_outputs)
        if total > 0:
            metrics.log_derived(
                "metrics/pretrain/accuracy",
                lambda meters: (
                    safe_round(
                        meters["pretrain/n_correct"].sum
                        * 100.0
                        / meters["pretrain/total"].sum,
                        3,
                    )
                    if meters["pretrain/total"].sum > 0
                    else float("nan")
                ),
            )
            metrics.log_derived(
                "metrics/pretrain/precision",
                lambda meters: (
                    safe_round(
                        meters["pretrain/tp"].sum
                        * 100.0
                        / (meters["pretrain/tp"].sum + meters["pretrain/fp"].sum),
                        3,
                    )
                    if meters["pretrain/tp"].sum + meters["pretrain/fp"].sum > 0
                    else float("nan")
                ),
            )
            metrics.log_derived(
                "metrics/pretrain/recall",
                lambda meters: (
                    safe_round(
                        meters["pretrain/tp"].sum
                        * 100.0
                        / (meters["pretrain/tp"].sum + meters["pretrain/fn"].sum),
                        3,
                    )
                    if meters["pretrain/tp"].sum + meters["pretrain/fn"].sum > 0
                    else float("nan")
                ),
            )
            metrics.log_derived(
                "metrics/pretrain/f1",
                lambda meters: (
                    safe_round(
                        meters["pretrain/tp"].sum
                        * 100.0
                        * 2
                        / (
                            2 * meters["pretrain/tp"].sum
                            + meters["pretrain/fn"].sum
                            + meters["pretrain/fp"].sum
                        ),
                        3,
                    )
                    if 2 * meters["pretrain/tp"].sum
                    + meters["pretrain/fn"].sum
                    + meters["pretrain/fp"].sum
                    > 0
                    else float("nan")
                ),
            )

            # print("\n from aggregating: \n", logging_outputs[0])
            for kk in ["_predictions", "_targets", "_source_size"]:
                if kk in logging_outputs[0]:
                    if kk == "_source_size":
                        metrics.log_scalar(kk, logging_outputs[0].get(kk, 0), round=1)
                    else:
                        metrics.log_custom(
                            ConcatTensorMeter,
                            kk,
                            torch.cat(
                                [logs[kk].cpu() for logs in logging_outputs], dim=0
                            ),
                        )
