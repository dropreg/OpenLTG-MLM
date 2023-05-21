# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from fairseq.models.transformer import TransformerModel, TransformerDecoder, TransformerEncoder
from .roberta_layers import TransformerSharedLayer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
from fairseq.modules.transformer_sentence_encoder import init_bert_params
import torch.nn.functional as F
from .iterative_refinement_generator import DecoderOut
from fairseq.utils import new_arange
import numpy as np


def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats

def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)

@register_model("long_roberta")
class LongRoberta(BaseFairseqModel):

    def __init__(self, args, encoder, decoder, span_sampling):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_dict =  decoder.dictionary

        self.bos = decoder.dictionary.bos()
        self.eos = decoder.dictionary.eos()
        self.pad = decoder.dictionary.pad()
        self.unk = decoder.dictionary.unk()
        self.mask_idx = decoder.dictionary.index("<mask>")

        self.share_encoder_decoder = args.share_encoder_decoder
        self.span_sampling = span_sampling
        
        if self.share_encoder_decoder:
            self.share_encoder_decoder_param()
        
    def share_encoder_decoder_param(self):
        self.encoder.embed_positions.weight = self.decoder.embed_positions.weight
        for enc_layer, dec_layer in zip(self.encoder.layers, self.decoder.layers):
            enc_layer.self_attn.k_proj.weight = dec_layer.self_attn.k_proj.weight
            enc_layer.self_attn.k_proj.bias = dec_layer.self_attn.k_proj.bias
            enc_layer.self_attn.v_proj.weight = dec_layer.self_attn.v_proj.weight
            enc_layer.self_attn.v_proj.bias = dec_layer.self_attn.v_proj.bias
            enc_layer.self_attn.q_proj.weight = dec_layer.self_attn.q_proj.weight
            enc_layer.self_attn.q_proj.bias = dec_layer.self_attn.q_proj.bias
            enc_layer.self_attn.out_proj.weight = dec_layer.self_attn.out_proj.weight
            enc_layer.self_attn.out_proj.bias = dec_layer.self_attn.out_proj.bias 
            enc_layer.self_attn_layer_norm.weight = dec_layer.self_attn_layer_norm.weight
            enc_layer.self_attn_layer_norm.bias = dec_layer.self_attn_layer_norm.bias
            enc_layer.fc1.weight = dec_layer.fc1.weight
            enc_layer.fc1.bias = dec_layer.fc1.bias
            enc_layer.fc2.weight = dec_layer.fc2.weight
            enc_layer.fc2.bias = dec_layer.fc2.bias
            enc_layer.final_layer_norm.weight = dec_layer.final_layer_norm.weight 
            enc_layer.final_layer_norm.bias = dec_layer.final_layer_norm.bias
        self.encoder.emb_layer_norm.weight = self.decoder.emb_layer_norm.weight
        self.encoder.emb_layer_norm.bias = self.decoder.emb_layer_norm.bias

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--checkpoint-activations', action='store_true',
                            help='checkpoint activations at each layer, which saves GPU '
                                 'memory usage at the cost of some additional compute')
        parser.add_argument('--offload-activations', action='store_true',
                            help='checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations.')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # fmt: on
        parser.add_argument("--apply-bert-init", action="store_true",
                            help="use custom param initialization for BERT",)
        
        # For long text generation
        parser.add_argument('--share-encoder-decoder', action='store_true',
                            help='share encoder and decoder parameters')
        parser.add_argument('--max-perdict-length', type=int, metavar='N',
                            help='max span sentence length')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder, task.span_sampling)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = RobertaEncoder(args, src_dict, embed_tokens)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = RobertaDecoder(args, tgt_dict, embed_tokens)
        return decoder

    def forward(
        self,
        src_segment, 
        src_span_length, 
        tgt_segment, 
        tgt_mask_segment, 
        tgt_span_length, 
    ):
        
        encoder_out = self.encoder(src_segment, src_span_length)
        
        output, length_out_list, length_tgt_list, extra_states = self.decoder(
            normalize=False,
            encoder_out=encoder_out,
            tgt_segment=tgt_segment,
            tgt_mask_segment=tgt_mask_segment,
            tgt_span_length=tgt_span_length,
        )
        return output, length_out_list, length_tgt_list, extra_states

    def forward_encoder(self, encoder_inputs):
        return self.encoder(*encoder_inputs)

    @property
    def allow_length_beam(self):
        return True

    def nucleus_sampling(self, probs, output_tokens, step, max_step):
        
        nucleus_p = 0.9
        nucleus_k = 100
        temperature = (1.0 - step / max_step) * 2.0
        probs = F.softmax(probs / temperature, dim=-1)
        raw_indices_buf = probs.max(-1)[1].unsqueeze(-1)
        
        if nucleus_p > 0:
            sorted_probs, sorted_indices = probs.sort(descending=True)
            cumsum_probs = sorted_probs.cumsum(dim=2)
            mask = cumsum_probs.lt(nucleus_p)

            cumsum_mask = mask.cumsum(dim=2)
            last_included = cumsum_mask[:, :, -1:]
            last_included.clamp_(0, mask.size()[2] - 1)
            mask = mask.scatter_(2, last_included, 1)
            
            max_dim = last_included.max()
            truncated_mask = mask[:, :, : max_dim + 1]
            truncated_probs = sorted_probs[:, :, : max_dim + 1]
            truncated_indices = sorted_indices[:, :, : max_dim + 1]
            trimed_probs = truncated_probs.masked_fill_(~truncated_mask, 0)
        else:
            trimed_probs, truncated_indices = probs.topk(nucleus_k)
        
        bsz, seq_len, _ = trimed_probs.size()
        select_buf = torch.multinomial(trimed_probs.view(bsz * seq_len, -1), 1, replacement=True).view(bsz, seq_len)
        scores_buf = torch.gather(trimed_probs, dim=2, index=select_buf.unsqueeze(-1))
        indices_buf = torch.gather(truncated_indices, dim=2, index=select_buf.unsqueeze(-1))
        
        # Filter unk token
        oov_mask = (indices_buf > 50260) | (indices_buf < 4)
        indices_buf.masked_scatter_(oov_mask, raw_indices_buf[oov_mask])
        
        return torch.log(scores_buf), indices_buf
        
    def forward_decoder(self, decoder_out, context_out, span_idx, decoding_format=None, **kwargs):
        step = decoder_out.step
        max_step = decoder_out.max_step
        
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history
        
        output_masks = output_tokens.eq(self.mask_idx)

        decoder_x, states, padding_mask = self.decoder.forward_inference(
            normalize=False,
            tgt_mask_segment=output_tokens,
            context_out=context_out,
            span_idx=span_idx,
            step=step,
        )
        
        # if step == 0 or step == 4:
        #     from fairseq import utils
        #     from fairseq.data.encoders import gpt2_bpe_utils
        #     from fairseq import file_utils
        #     encoder_json = file_utils.cached_path("/opt/data/private/code/fairseq/examples_acl/long2long_bert/scripts/common/encoder.json")
        #     vocab_bpe = file_utils.cached_path("/opt/data/private/code/fairseq/examples_acl/long2long_bert/scripts/common/vocab.bpe")
        #     bpe = gpt2_bpe_utils.get_encoder(encoder_json, vocab_bpe)
        #     def bpe_decode(x):
        #         return bpe.decode(
        #             [int(tok) if tok not in {"<unk>", "<pad>", "<mask>"} else tok for tok in x.split()]
        #         )
            
        #     for pos in [40, 80]:
        #         for v in torch.topk(F.softmax(decoder_x, dim=-1), 100)[1][0, pos, :50]:
        #             if v > 3:
        #                 print("'" + bpe.decode([int(self.tgt_dict[v])]) + "',")
        #         for k in torch.topk(F.softmax(decoder_x, dim=-1), 100)[0][0, pos, :50]:
        #             print(k.cpu().numpy())
        #         print("---" * 20, step)
        # import pdb; pdb.set_trace()

        # torch.topk(F.softmax(decoder_x, dim=-1), 100)[1][0, 119, -20:]
        # import pdb; pdb.set_trace()
        
        if step < max_step and self.span_sampling:
            _scores, _tokens = self.nucleus_sampling(decoder_x, output_tokens, step, max_step)
        else:
            _scores, _tokens = F.log_softmax(decoder_x, -1).max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        
        if history is not None:
            history.append(output_tokens.clone())
        
        history_token = output_tokens.clone()
        # skeptical decoding (depend on the maximum decoding steps.) 
        if (step + 1) < max_step:
            skeptical_mask = _skeptical_unmasking(
                output_scores, output_tokens.ne(self.pad),  1 - (step + 1) / max_step, 
            )
            output_tokens.masked_fill_(skeptical_mask, self.mask_idx)
            
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

            output_masks = skeptical_mask
        
        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
            input_mask=output_masks,
        ), states, padding_mask, history_token
        
    def initialize_output_tokens(self, sample, encoder_out, src_tokens):
        
        # length prediction
        enc_feats = torch.cat(encoder_out["encoder_out"], dim=0)
        src_masks = torch.cat(encoder_out["encoder_padding_mask"], dim=-1)
        length_out = self.decoder.forward_length(True, enc_feats, src_masks)
        length_tgt = self.decoder.forward_length_prediction(length_out, enc_feats, src_masks)

        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)

        initial_output_tokens_mask = idx_length[None, :] < length_tgt[:, None]
        initial_output_tokens.masked_fill_(
            initial_output_tokens_mask, self.mask_idx
        )
        
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)
        initial_output_tokens_mask[:, 0] = False
        initial_output_tokens_mask.scatter_(1, length_tgt[:, None] - 1, False)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        # nopad_mask = sample['target'].ne(self.bos) & sample['target'].ne(self.eos) & sample['target'].ne(self.pad)
        # random_masking_indices = torch.bernoulli(torch.full(sample['target'].shape, 0.97, device=sample['target'].device)).bool()
        # initial_output_tokens = sample['target'].clone()
        # masking_indices = random_masking_indices & nopad_mask
        # initial_output_tokens[masking_indices] = self.mask_idx
        # initial_output_tokens_mask = masking_indices
        # initial_output_scores = initial_output_tokens.new_zeros(
        #     *initial_output_tokens.size()
        # ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
            input_mask=initial_output_tokens_mask,
        )

    def initialize_iter_output_tokens(self, context_state_dict, src_tokens):
        
        enc_feats = torch.cat(context_state_dict["states"], dim=0)
        src_masks = torch.cat(context_state_dict["padding_mask"], dim=-1)

        length_out = self.decoder.forward_length(True, enc_feats, src_masks)
        length_tgt = self.decoder.forward_length_prediction(length_out, enc_feats, src_masks)
        
        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)

        initial_output_tokens_mask = idx_length[None, :] < length_tgt[:, None]
        initial_output_tokens.masked_fill_(
            initial_output_tokens_mask, self.mask_idx
        )
        
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)
        initial_output_tokens_mask[:, 0] = False
        initial_output_tokens_mask.scatter_(1, length_tgt[:, None] - 1, False)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(enc_feats)

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
            input_mask=initial_output_tokens_mask,
        )

    def regenerate_length_beam(self, decoder_out, beam_size):
        
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        #  beam_size // 2
        length_tgt = (
            length_tgt[:, None]
            + utils.new_arange(length_tgt, 1, beam_size)
            - beam_size // 2
        )
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        
        initial_output_tokens_mask = idx_length[None, :] < length_tgt[:, None]
        initial_output_tokens.masked_fill_(
            initial_output_tokens_mask, self.mask_idx
        )

        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_tokens_mask[:, 0] = False
        initial_output_tokens_mask.scatter_(1, length_tgt[:, None] - 1, False)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, 
            output_scores=initial_output_scores,
            input_mask=initial_output_tokens_mask,
        )
    
    def upgrade_state_dict_named(self, state_dict, name):
        

        for k in list(state_dict.keys()):
            
            if "embed_length" in k and state_dict["decoder.embed_length.weight"].size(0) != self.decoder.embed_length.weight.size(0):
                state_dict["decoder.embed_length.weight"] = self.decoder.embed_length.weight
                
            if "sentence_encoder.embed_tokens.weight" in k:
                state_dict['encoder.embed_tokens.weight'] = state_dict[k]
                state_dict['decoder.embed_tokens.weight'] = state_dict[k].clone()
                del state_dict[k]
                
                # self defined
                state_dict["decoder.embed_length.weight"] = self.decoder.embed_length.weight

            elif "sentence_encoder.embed_positions.weight" in k:
                state_dict['encoder.embed_positions.weight'] = state_dict[k]
                state_dict['decoder.embed_positions.weight'] = state_dict[k].clone()
                del state_dict[k]
            
            elif "sentence_encoder.layers." in k:
                # import pdb; pdb.set_trace()
                # if int(k[32]) < 6 and "10" not in k and "11" not in k:
                replace_encoder_key = k.replace("decoder.sentence_encoder", "encoder")
                replace_decoder_key = k.replace("decoder.sentence_encoder", "decoder")
                state_dict[replace_encoder_key] = state_dict[k]
                state_dict[replace_decoder_key] = state_dict[k].clone()
                del state_dict[k]
            elif "sentence_encoder.emb_layer_norm" in k:
                if "weight" in k:
                    state_dict["encoder.emb_layer_norm.weight"] = state_dict[k]
                    state_dict["decoder.emb_layer_norm.weight"] = state_dict[k].clone()
                if "bias" in k:
                    state_dict["encoder.emb_layer_norm.bias"] = state_dict[k]
                    state_dict["decoder.emb_layer_norm.bias"] = state_dict[k].clone()
                del state_dict[k]
            
            if 'encoder.sentence_encoder.version' in k:
                del state_dict[k]
        super().upgrade_state_dict_named(state_dict, name)


class RobertaEncoder(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        
        self.embed_dim = embed_tokens.embedding_dim
        self.embed_positions = (
            PositionalEmbedding(
                512,
                self.embed_dim,
                padding_idx=self.padding_idx,
                learned=True,
            )
        )
        self.emb_layer_norm = LayerNorm(self.embed_dim)

    def build_encoder_layer(self, args):
        layer = TransformerSharedLayer(args)
        return layer

    def forward_embedding(self, src_segment, src_span_length):
        
        span_list = []
        span_embed_list = []
        span_padding_list = []
        span_start = 0

        for span_idx, span_len in enumerate(src_span_length):
            src_tokens = src_segment[:, span_start: span_start+span_len]
            
            padding_mask = src_tokens.eq(self.padding_idx)
            token_embedding = self.embed_tokens(src_tokens)
            x = embed = token_embedding
            if self.embed_positions is not None:
                x = embed + self.embed_positions(src_tokens)
            x = self.emb_layer_norm(x)
            x = self.dropout_module(x)
            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

            span_list.append(x)
            span_embed_list.append(embed)
            span_padding_list.append(padding_mask)
            span_start += span_len
            
        return span_list, span_embed_list, span_padding_list

    def forward(
        self,
        src_segment, 
        src_span_length,
    ):
        span_list, span_embed_list, span_padding_list = self.forward_embedding(src_segment, src_span_length)

        context_states_list = []
        context_padding_mask_list = []
        for span_x, span_padding_mask in zip(span_list, span_padding_list):
            
            x = span_x.transpose(0, 1)
            encoder_padding_mask = span_padding_mask
            
            if len(context_states_list) == 0:
                for layer in self.layers:
                    x = layer.forward_encoder(x, encoder_padding_mask)
            else:

                context_x = torch.cat(context_states_list, dim=0)
                context_padding_mask = torch.cat(context_padding_mask_list, dim=-1) 

                for i, layer in enumerate(self.layers):    
                    x = layer.forward_decoder(
                        x,
                        context_x,
                        encoder_padding_mask=context_padding_mask,
                        decoder_padding_mask=encoder_padding_mask,
                    )

            context_states_list.append(x)
            context_padding_mask_list.append(span_padding_mask)
        
        return {
            "encoder_out": context_states_list,
            "encoder_padding_mask": context_padding_mask_list,
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [],
        }

class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x


class RobertaDecoder(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        
        self.embed_positions = (
            PositionalEmbedding(
                512,
                self.embed_dim,
                padding_idx=self.padding_idx,
                learned=True,
            )
        )
        self.emb_layer_norm = LayerNorm(self.embed_dim)

        self.lm_head = self.build_lm_head(
            embed_dim=self.embed_dim,
            output_dim=len(dictionary),
            activation_fn="gelu",
            weight=embed_tokens.weight,
        )
        self.output_projection = None
        
        
        self.bos = dictionary.bos()
        self.eos = dictionary.eos()
        self.mask_idx = dictionary.index("<mask>")

        self.max_perdict_length = args.max_perdict_length
        self.embed_length = Embedding(self.max_perdict_length, self.embed_dim, None)
        self.embed_length.weight.data.normal_(mean=0.0, std=0.02)
        
    def build_lm_head(self, embed_dim, output_dim, activation_fn, weight):
        return RobertaLMHead(embed_dim, output_dim, activation_fn, weight)
    
    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerSharedLayer(args)
        return layer

    def output_layer(self, features):
        return self.lm_head(features)

    def forward(
        self,
        normalize,
        encoder_out,     
        tgt_segment,
        tgt_mask_segment,
        tgt_span_length,
        step=0, 
        **unused
    ):
        features, len_out, len_tgt, extra_states = self.extract_features(
            encoder_out=encoder_out,
            tgt_segment=tgt_segment,
            tgt_mask_segment=tgt_mask_segment,
            tgt_span_length=tgt_span_length,
            step=step,
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out, len_out, len_tgt, extra_states

    def forward_length(self, normalize, enc_feats, src_masks):
        enc_feats = _mean_pooling(enc_feats, src_masks)
        enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out

    def forward_length_prediction(self, length_out, enc_feats, src_masks, tgt_tokens=None):
        
        if tgt_tokens is not None:
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            length_tgt = tgt_lengs.clamp(min=0, max=self.max_perdict_length - 1)
        else:
            pred_lengs = length_out.max(-1)[1]
            length_tgt = pred_lengs
            length_tgt[:] = 140
            # length_tgt[:] = 154
            # length_tgt[:] -= 20
            # length_tgt[:] += 5
            # length_tgt[:] -= 3

        return length_tgt

    def forward_span_embedding(self, tgt_segment, tgt_span_length):

        span_token_list = []
        span_list = []
        span_padding_list = []
        span_start = 0

        for span_idx, span_len in enumerate(tgt_span_length):
            
            span_tokens = tgt_segment[:, span_start: span_start + span_len]
            span_token_list.append(span_tokens)

            x = self.embed_tokens(span_tokens)
            x += self.embed_positions(span_tokens)

            x = self.emb_layer_norm(x)
            x = self.dropout_module(x)
            if self.quant_noise is not None:
                x = self.quant_noise(x)
            
            padding_mask = span_tokens.eq(self.padding_idx)
            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

            span_list.append(x)
            span_padding_list.append(padding_mask)
            span_start += span_len

        return span_token_list, span_list, span_padding_list

    def extract_features(
        self,
        encoder_out=None,
        tgt_segment=None,
        tgt_mask_segment=None,
        tgt_span_length=None,
        step=None,
        **unused
    ):
        
        span_mask_tokens_list, span_mask_list, span_mask_padding_list = self.forward_span_embedding(tgt_mask_segment, tgt_span_length)
        span_tokens_list, span_list, _ = self.forward_span_embedding(tgt_segment, tgt_span_length)
        
        # Build Context
        context_states_list = encoder_out["encoder_out"]
        context_padding_mask_list = encoder_out["encoder_padding_mask"]

        lengt_out_list = []
        length_tgt_list = []

        decoder_out = []
        for mask_x, span_x, span_target, span_padding_mask in zip(span_mask_list, span_list, span_tokens_list, span_mask_padding_list):
            
            x = None
            t_x = None
            context_x = torch.cat(context_states_list, dim=0)
            context_padding_mask = torch.cat(context_padding_mask_list, dim=-1) 

            for i, layer in enumerate(self.layers):
                
                if x is None:
                    x = mask_x
                    x = x.transpose(0, 1)
                    t_x = span_x
                    t_x = t_x.transpose(0, 1)
                
                x = layer.forward_decoder(
                    x,
                    context_x,
                    encoder_padding_mask=context_padding_mask,
                    decoder_padding_mask=span_padding_mask,
                )
                
                t_x = layer.forward_decoder(
                    t_x,
                    context_x,
                    encoder_padding_mask=context_padding_mask,
                    decoder_padding_mask=span_padding_mask,
                )
            
            # For length prediction
            length_out = self.forward_length(False, context_x, context_padding_mask)
            length_tgt = self.forward_length_prediction(length_out, context_x, context_padding_mask, span_target)
            lengt_out_list.append(length_out)
            length_tgt_list.append(length_tgt)

            x = x.transpose(0, 1)
            decoder_out.append(x)
            
            context_states_list.append(t_x)
            context_padding_mask_list.append(span_padding_mask)

        features = torch.cat(decoder_out, dim=1)
        return features, lengt_out_list, length_tgt_list, {"states": [], "pad": []}
        
    def forward_embedding(self, src_tokens, context_padding_mask_list, span_idx):

        padding_mask = src_tokens.eq(self.padding_idx)
        token_embedding = self.embed_tokens(src_tokens)
        x = token_embedding
        if self.embed_positions is not None:
            x = x + self.embed_positions(src_tokens)

        x = self.emb_layer_norm(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        return x, padding_mask

    def forward_inference(self, normalize, tgt_mask_segment, context_out, span_idx, step=0, **unused):
        features, states, padding_mask = self.extract_features_inference(
            tgt_mask_segment=tgt_mask_segment,
            context_out=context_out,
            span_idx=span_idx,
            step=step,
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out, states, padding_mask

    def extract_features_inference(
        self,
        tgt_mask_segment=None,
        context_out=None,
        span_idx=None,
        step=0,
        **unused
    ):
        x, padding_mask = self.forward_embedding(tgt_mask_segment, context_out["padding_mask"], span_idx)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        
        context_states_list = context_out["states"]
        context_padding_mask_list = context_out["padding_mask"]

        context_x = torch.cat(context_states_list, dim=0)
        context_padding_mask = torch.cat(context_padding_mask_list, dim=-1) 

        for i, layer in enumerate(self.layers):
            
            x = layer.forward_decoder(
                x,
                context_x,
                encoder_padding_mask=context_padding_mask,
                decoder_padding_mask=padding_mask,
            )
        states = x
        x = x.transpose(0, 1)
        return x, states, padding_mask


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("long_roberta", "long_roberta")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

@register_model_architecture("long_roberta", "long_roberta_large")
def large_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_architecture(args)
