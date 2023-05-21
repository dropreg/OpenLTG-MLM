# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple

import numpy as np
import torch
from fairseq import utils
from fairseq.data.encoders import gpt2_bpe_utils
from fairseq import file_utils

DecoderOut = namedtuple(
    "IterativeRefinementDecoderOut",
    ["output_tokens", "output_scores", "keep_mask", "attn", "step", "max_step", "history"],
)


class PlanIterativeRefinement(object):
    def __init__(
        self,
        tgt_dict,
        models=None,
        eos_penalty=0.0,
        max_iter=10,
        max_ratio=2,
        beam_size=1,
        decoding_format=None,
        retain_dropout=False,
        adaptive=True,
        retain_history=False,
        reranking=False,
    ):
        """
        Generates translations based on iterative refinement.

        Args:
            tgt_dict: target dictionary
            eos_penalty: if > 0.0, it penalized early-stopping in decoding
            max_iter: maximum number of refinement iterations
            max_ratio: generate sequences of maximum length ax, where x is the source length
            decoding_format: decoding mode in {'unigram', 'ensemble', 'vote', 'dp', 'bs'}
            retain_dropout: retaining dropout in the inference
            adaptive: decoding with early stop
        """
        self.bos = tgt_dict.bos()
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.eos_penalty = eos_penalty
        self.max_iter = max_iter
        self.max_ratio = max_ratio
        self.beam_size = beam_size
        self.reranking = reranking
        self.decoding_format = decoding_format
        self.retain_dropout = retain_dropout
        self.retain_history = retain_history
        self.adaptive = adaptive
        self.models = models
        self.tgt_dict = tgt_dict

    def generate_batched_itr(
        self,
        data_itr,
        maxlen_a=None,
        maxlen_b=None,
        cuda=False,
        timer=None,
        prefix_size=0,
    ):
        """Iterate over a batched dataset and yield individual translations.

        Args:
            maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda: use GPU for generation
            timer: StopwatchMeter for timing generations.
        """

        for sample in data_itr:
            if "net_input" not in sample:
                continue
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(
                    self.models,
                    sample,
                    prefix_tokens=sample["target"][:, :prefix_size]
                    if prefix_size > 0
                    else None,
                )
            if timer is not None:
                timer.stop(sample["ntokens"])
            for i, id in enumerate(sample["id"]):
                # remove padding
                src = utils.strip_pad(sample["net_input"]["src_tokens"][i, :], self.pad)
                ref = utils.strip_pad(sample["target"][i, :], self.pad)
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, models, sample, prefix_tokens=None, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the IterativeRefinementGenerator is not supported"
            )

        # TODO: iterative refinement generator does not support ensemble for now.
        if not self.retain_dropout:
            for model in models:
                model.eval()

        model, reranker = models[0], None
        if self.reranking:
            assert len(models) > 1, "Assuming the last checkpoint is the reranker"
            assert (
                self.beam_size > 1
            ), "Reranking requires multiple translation for each example"

            reranker = models[-1]
            models = models[:-1]

        if len(models) > 1 and hasattr(model, "enable_ensemble"):
            assert model.allow_ensemble, "{} does not support ensembling".format(
                model.__class__.__name__
            )
            model.enable_ensemble(models)

        # TODO: better encoder inputs?
        src_tokens = sample["net_input"]["src_segment"]
        bsz, src_len = src_tokens.size()

        # initialize
        encoder_out = model.forward_encoder([src_tokens])
        decoder_out = model.initialize_output_tokens(sample, encoder_out, src_tokens)

        if self.beam_size > 1:
            assert (
                model.allow_length_beam
            ), "{} does not support decoding with length beam.".format(
                model.__class__.__name__
            )

            # regenerate data based on length-beam
            length_beam_order = (
                utils.new_arange(src_tokens, self.beam_size, bsz).t().reshape(-1)
            )
            encoder_out = model.encoder.reorder_encoder_out(
                encoder_out, length_beam_order
            )
            decoder_out = model.regenerate_length_beam(
                decoder_out, self.beam_size
            )
            bsz = bsz * self.beam_size

        sent_idxs = torch.arange(bsz)
        output_tokens = decoder_out.output_tokens.clone()

        if self.retain_history:
            decoder_out = decoder_out._replace(history=[decoder_out])

        finalized = [[] for _ in range(bsz)]

        def finalized_hypos(step, prev_out_token, prev_out_score, prev_out_attn):
            cutoff = prev_out_token.ne(self.pad)
            tokens = prev_out_token[cutoff]
            if prev_out_score is None:
                scores, score = None, None
            else:
                scores = prev_out_score[cutoff]
                score = scores.mean()

            if prev_out_attn is None:
                hypo_attn, alignment = None, None
            else:
                hypo_attn = prev_out_attn[cutoff]
                alignment = hypo_attn.max(dim=1)[1]
            
            return {
                "steps": step,
                "tokens": tokens,
                "positional_scores": scores,
                "score": score,
                "hypo_attn": hypo_attn,
                "alignment": alignment,
            }
        
        # encoder_json = file_utils.cached_path("/opt/data/private/code/fairseq/examples/long2long_bert/scripts/common/encoder.json")
        # vocab_bpe = file_utils.cached_path("/opt/data/private/code/fairseq/examples/long2long_bert/scripts/common/vocab.bpe")
        # bpe = gpt2_bpe_utils.get_encoder(encoder_json, vocab_bpe)
        # def bpe_decode(x):
        #     return bpe.decode(
        #         [int(tok) if tok not in {"<unk>", "<pad>", "<mask>"} else tok for tok in x.split()]
        #     )
        
        for step in range(self.max_iter + 1):
            
            decoder_options = {
                "eos_penalty": self.eos_penalty,
                "max_ratio": self.max_ratio,
                "decoding_format": self.decoding_format,
                "history": decoder_out.output_tokens,
            }
            decoder_out = decoder_out._replace(
                step=step,
                max_step=self.max_iter + 1,
            )
            decoder_out = model.forward_decoder(
                decoder_out, encoder_out, **decoder_options
            )

        for i in range(sent_idxs.size(0)):
            finalized[sent_idxs[i]] = [
                finalized_hypos(
                    step,
                    decoder_out.output_tokens[i],
                    decoder_out.output_scores[i],
                    None,
                )
            ]
        
        if self.beam_size > 1:
            # aggregate information from length beam
            finalized = [
                finalized[
                    np.argmax(
                        [
                            finalized[self.beam_size * i + j][0]["score"].cpu()
                            for j in range(self.beam_size)
                        ]
                    )
                    + self.beam_size * i
                ]
                for i in range(len(finalized) // self.beam_size)
            ]

        return finalized

