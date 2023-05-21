# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import torch.nn.functional as F
import random

@register_criterion("span_label_smooth_loss")
class SpanLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(
        self,
        task,
    ):
        super().__init__(task)
        self.eps = 0.1
    
    def build_input_data(self, sample):
        model_input ={
            "src_segment": sample["net_input"]["src_segment"],
            "src_span_length": sample["net_input"]["src_span_length"],
            "tgt_segment": sample["net_input"]["tgt_segment"],
            "tgt_mask_segment": sample["net_input"]["tgt_mask_segment"],
            "tgt_span_length": sample["net_input"]["tgt_span_length"],
        }
        return model_input
        
    def forward(self, model, sample, reduce=True):
        
        net_input = self.build_input_data(sample)
        output, length_out_list, length_tgt_list, _ = model(**net_input)
        
        target = sample["net_input"]["tgt_segment"]
        mask = sample["net_input"]["tgt_mask_segment"].ne(target)
        
        loss, nll_loss = self.label_smooth_loss(output[mask], target[mask])
        length_loss = 0
        
        for length_out, length_tgt in zip(length_out_list, length_tgt_list):
            select_mask = length_tgt != 0
            if select_mask.sum(0) < select_mask.size(0):
                continue
            length_loss += self.length_loss(length_out, length_tgt)

        loss += 0.1 * length_loss
        
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        } 
        return loss, sample_size, logging_output

    def label_smooth_loss(self, net_out, net_target):
        net_logits = F.log_softmax(net_out, dim=-1)
        nll_loss = F.nll_loss(net_logits, net_target, reduction="none").float().mean()
        loss = nll_loss * (1. - self.eps) - net_logits.float().mean() * self.eps
        return loss, nll_loss

    def length_loss(self, length_out, length_tgt):
        length_logits = F.log_softmax(length_out, dim=-1)
        length_loss = F.nll_loss(length_logits, length_tgt, reduction="none").float().mean()
        return length_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
