# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils
from fairseq.utils import new_arange
import math
from fairseq import utils
from fairseq.data.encoders import gpt2_bpe_utils
from fairseq import file_utils

logger = logging.getLogger(__name__)

def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if (
            alignment[:, 0].max().item() >= src_len - 1
            or alignment[:, 1].max().item() >= tgt_len - 1
        ):
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True
        )
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    
    def merge_data(data_name, sort_order):
        prepared_data = merge(
            data_name,
            left_pad=left_pad_target,
            pad_to_length=pad_to_length[data_name]
            if pad_to_length is not None
            else None,
        )
        return prepared_data.index_select(0, sort_order)

    ############################################################
    src_segment = None
    src_cls = None
    tgt_segment = None
    
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()
        src_segment = merge(
            "src_segment",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["src_segment"]
            if pad_to_length is not None
            else None,
        )
        src_segment = src_segment.index_select(0, sort_order)
        tgt_segment = merge(
            "tgt_segment",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["tgt_segment"]
            if pad_to_length is not None
            else None,
        )
        tgt_segment = tgt_segment.index_select(0, sort_order)
        tgt_mask = merge(
            "tgt_mask",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["tgt_mask"]
            if pad_to_length is not None
            else None,
        )
        tgt_mask = tgt_mask.index_select(0, sort_order)
        ############################################################
    else:
        ntokens = src_lengths.sum().item()
        
    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "src_segment": src_segment,
            "tgt_segment": tgt_segment,
            "tgt_mask": tgt_mask,
        },
        "target": target,
    }

    if samples[0].get("alignment", None) is not None:
        bsz, tgt_sz = batch["target"].shape
        src_sz = batch["net_input"]["src_tokens"].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += torch.arange(len(sort_order), dtype=torch.long) * tgt_sz
        if left_pad_source:
            offsets[:, 0] += src_sz - src_lengths
        if left_pad_target:
            offsets[:, 1] += tgt_sz - tgt_lengths

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(
                sort_order, offsets, src_lengths, tgt_lengths
            )
            for alignment in [samples[align_idx]["alignment"].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch["alignments"] = alignments
            batch["align_weights"] = align_weights

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        lens = [sample.get("constraints").size(0) for sample in samples]
        max_len = max(lens)
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0 : lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints

    return batch


class LanguagePairDataset(FairseqDataset):
    
    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        plan=None,
        plan_sizes=None,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(
                tgt
            ), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.plan = plan
        self.plan_sizes = plan_sizes

        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info(
                    "bucketing target lengths: {}".format(list(self.tgt.buckets))
                )

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple

        encoder_json = file_utils.cached_path("/opt/data/private/code/fairseq/examples_acl/long2long_bert/scripts/common/encoder.json")
        vocab_bpe = file_utils.cached_path("/opt/data/private/code/fairseq/examples_acl/long2long_bert/scripts/common/vocab.bpe")
        self.bpe = gpt2_bpe_utils.get_encoder(encoder_json, vocab_bpe)

    def bpe_decode(self, x):
        return self.bpe.decode([int(tok) if tok not in {"<unk>", "<pad>", "<mask>"} else tok for tok in x.split()])
        
    def bpe_is_start(self, x):
        return [self.bpe.decode([int(tok)]).startswith(" ") if tok not in {"<unk>", "<pad>", "<mask>"} else tok for tok in x.split()]
    
    def find_index(self, key_words, sequence, seq_start):
        
        seq_str = self.bpe_decode(sequence)
        seq_whole_map = {}
        start = 0
        # map sequence.split() to seq_str.split()
        for whole_idx, whole_num in enumerate(np.array(seq_start).cumsum() - 1):
            if whole_num not in seq_whole_map:
                seq_whole_map[whole_num] = start
            start += 1
        
        seq_map = {}
        start = 0
        # map seq_str.split() to seq_str
        for seq_idx, seq_word in enumerate(seq_str.split()):
            for _ in seq_word:
                seq_map[start] = seq_idx
                start += 1
            start += 1
        
        keyphrase_list = []
        keyphrase_start_list = []
        keyphrase_end_list = []
        # SPLIT BY < s >
        for key_word in key_words.split("1279 264 1875"):
            key_str = self.bpe_decode(key_word).strip()
            keyphrase = [self.tgt_dict.index(k) for k in key_word.strip().split()]
            
            # find idx in seq_str
            f_start = 0
            while f_start < len(seq_str):
                idx = seq_str.find(key_str, f_start)
                if  idx <= 0 or (idx + len(key_str) == len(seq_str)) or (seq_str[idx - 1] == " " and seq_str[idx + len(key_str)] == " "):
                    break
                f_start = idx + 1
            
            if idx == -1:
                continue

            # find idx in seq_str.split()
            split_idx = seq_map[idx]
            # find split_idx in sequence.split()
            key_start = seq_whole_map[split_idx]
            if idx + len(key_str) == len(seq_str):
                key_end = len(seq_str) - 1
            else:
                key_end = seq_whole_map[split_idx + len(key_str.split())]

            if key_str != self.bpe_decode(" ".join(sequence.split()[key_start:key_end])).strip():
                continue
            
            keyphrase_list.append(keyphrase)
            keyphrase_start_list.append(key_start)
            keyphrase_end_list.append(key_end)
        return keyphrase_list, keyphrase_start_list, keyphrase_end_list


    def get_batch_shapes(self):
        return self.buckets
        
    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        plan_item = self.plan[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])
            
            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])
        
        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]
        
        # find target index
        tgt_start_list = self.bpe_is_start(self.tgt_dict.string(tgt_item.numpy()))
        tgt_start_list[0] = True

        plan_item_idx = self.tgt_dict.string(plan_item.numpy())
        tgt_item_idx = self.tgt_dict.string(tgt_item.numpy())
        keyphrase_list, keyphrase_start, keyphrase_end = self.find_index(plan_item_idx, tgt_item_idx, tgt_start_list)

        eos = self.tgt_dict.eos()
        kp_list = [torch.LongTensor([eos])]
        tgt_mask = torch.zeros(tgt_item.size())
        for kp, kp_start, kp_end  in zip(keyphrase_list, keyphrase_start, keyphrase_end):
            kp_list.append(torch.LongTensor(kp))
            kp_list.append(torch.LongTensor([eos]))
            tgt_mask[kp_start+1:kp_end+1] = 1

        keyphrase_item = torch.cat(kp_list)
        src_segment = torch.cat([src_item, keyphrase_item])

        # self.bpe_decode(self.tgt_dict.string(tgt_item[tgt_mask.bool()].numpy()))
        if tgt_item is not None:
            tgt_start_list = torch.Tensor([True] + tgt_start_list + [True]).bool()
            tgt_segment = self.build_mask_tokens(tgt_item.clone(), tgt_mask.bool())
        else:
            tgt_segment = None
        
        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
            "src_segment": src_segment,
            "tgt_segment": tgt_segment,
            "tgt_mask": tgt_mask.bool(),
        }
        
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example

    def build_mask_tokens(self, target_tokens, keep_mask):
        pad = self.tgt_dict.pad()
        bos = self.tgt_dict.bos()
        eos = self.tgt_dict.eos()
        mask = self.tgt_dict.index('<mask>')

        target_masks = (target_tokens.ne(bos) & target_tokens.ne(eos))
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(~target_masks, 2.0)
        target_length = target_masks.sum().float()
        target_length = target_length * target_length.clone().uniform_()
        target_length = target_length + 1  # make sure to mask at least one token.
        _, target_rank = target_score.sort(0)
        target_cutoff = new_arange(target_rank) < target_length.long()
        target_mask = target_cutoff.scatter(0, target_rank, target_cutoff)
        mask_token = target_tokens.clone().masked_fill(target_mask & ~keep_mask, mask)
        return mask_token
        
    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )
