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
from .plan_iterative_refinement import DecoderOut
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

@register_model("plan_roberta")
class PlanRoberta(BaseFairseqModel):
    
    def __init__(self, args, encoder, decoder, decoding_sampling):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = decoder

        self.bos = decoder.dictionary.bos()
        self.eos = decoder.dictionary.eos()
        self.pad = decoder.dictionary.pad()
        self.unk = decoder.dictionary.unk()
        self.mask_idx = decoder.dictionary.index("<mask>")

        self.share_encoder_decoder = args.share_encoder_decoder
        self.decoding_sampling = decoding_sampling
        
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
        return cls(args, encoder, decoder, task.decoding_sampling)

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
        tgt_segment,
        target,
    ):
        encoder_out = self.encoder(src_segment)
        length_out = self.decoder.forward_length(
            False,
            encoder_out["encoder_out"][0],
            encoder_out["encoder_padding_mask"][0],
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out,             
            encoder_out["encoder_out"][0],
            encoder_out["encoder_padding_mask"][0],
            target,
        )
        output = self.decoder(
            normalize=False,
            encoder_out=encoder_out,     
            tgt_segment=tgt_segment,
        )
        return output, length_out, length_tgt


    def forward_encoder(self, encoder_inputs):
        return self.encoder(*encoder_inputs)

    @property
    def allow_length_beam(self):
        return True

    def decay_func(self, x):
        return 2.0 * np.exp( -(x ** 0.5))

    def nucleus_sampling(self, probs, output_tokens, step, max_step):
        
        nucleus_p = 0.9
        nucleus_k = 100
        # temperature = self.decay_func(step)
        temperature = 1.0
        temperature = (1.0 - step / max_step) * 1.5
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
        oov_mask = (indices_buf > 50260)
        indices_buf.masked_scatter_(oov_mask, raw_indices_buf[oov_mask])
        
        return torch.log(scores_buf), indices_buf


    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        max_step = decoder_out.max_step
        
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history
        
        keep_masks = decoder_out.keep_mask
        output_masks = output_tokens.eq(self.mask_idx)

        decoder_x = self.decoder(
            normalize=False,
            encoder_out=encoder_out,     
            tgt_segment=output_tokens,
            step=step,
        )
        
        if self.decoding_sampling:
            _scores, _tokens = self.nucleus_sampling(decoder_x, output_tokens, step, max_step)
        else:
            _scores, _tokens = F.log_softmax(decoder_x, -1).max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        
        if history is not None:
            history.append(output_tokens.clone())
        
        # skeptical decoding (depend on the maximum decoding steps.) 
        if (step + 1) < max_step:
            skeptical_mask = _skeptical_unmasking(
                output_scores, output_tokens.ne(self.pad),  1 - (step + 1) / max_step, 
            )

            output_tokens.masked_fill_(skeptical_mask & ~keep_masks, self.mask_idx)
            output_scores.masked_fill_(skeptical_mask & ~keep_masks, 0.0)
            output_scores.masked_fill_(keep_masks, 1.0)

            if history is not None:
                history.append(output_tokens.clone())
            
        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )
        
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
        initial_output_tokens.masked_fill_(initial_output_tokens_mask, self.mask_idx)
        
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)
        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])
        
        # fill the token
        # src_mask = sample["net_input"]["src_cls"] > 1
        # keyphrase_length = self.encoder.forward_length(encoder_out["encoder_out"][0].transpose(0, 1))
        # keyphrase_score = F.softmax(keyphrase_length, dim=-1)
        
        # bsz, seq_len = sample["net_input"]["src_cls"].size()
        # for b in range(bsz):
        #     start_flag = False
        #     token_list = []
        #     all_token_list = []
        #     pos_list = []
        #     score_list = []
        #     for s in range(seq_len):
                
        #         if sample["net_input"]["src_cls"][b][s] != 0:
        #             if not start_flag:
        #                 start_flag = True
        #             token_list.append(sample["net_input"]["src_segment"][b][s])
        #             score_list.append(keyphrase_score[b][s].unsqueeze(0))
        #         else:
        #             if start_flag:
        #                 start_flag = False
        #                 pos = torch.cat(score_list, dim=0).mean(0).max(-1)[1]
        #                 pos_list.append(pos)
        #                 all_token_list.append(token_list)
        #                 token_list = []
        #     pos_list.sort()
        #     start = min(pos_list)
        #     offset = (length_tgt[0] - 2 * start) // len(pos_list)
        #     for j in range(len(pos_list)):
        #         for token_idx, token in enumerate(all_token_list[j]):
        #             initial_output_tokens[b][start + offset * j + token_idx] = token
            
        #     score_list = []
        #     all_token_list = []
        #     pos_list = []
        
        # nopad_mask = sample["target"].clone().eq(self.pad)
        # initial_output_tokens = sample["target"].clone().masked_fill(~sample["net_input"]["tgt_mask"], self.mask_idx)
        # initial_output_tokens[:, 0] = self.bos
        # length_tgt = sample["target"].clone().ne(self.pad).sum(-1)
        # initial_output_tokens.masked_fill(nopad_mask, self.pad)

        # initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)
        # initial_output_scores = initial_output_tokens.new_zeros(
        #     *initial_output_tokens.size()
        # ).type_as(encoder_out["encoder_out"][0])
        # initial_output_tokens_mask = sample["net_input"]["tgt_mask"]
        
        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
            keep_mask=initial_output_tokens_mask,
        )

    def regenerate_length_beam(self, decoder_out, beam_size):
        
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
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
        if self.random_corrupt:
            random_token = np.random.choice(len(self.vocab_probs), initial_output_tokens.size(), p=self.vocab_probs)
            random_token = torch.from_numpy(random_token).to(initial_output_tokens.device)
            initial_output_tokens = initial_output_tokens.masked_scatter_(initial_output_tokens_mask, random_token[initial_output_tokens_mask])
        else:
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
            keep_mask=initial_output_tokens_mask,
        )
    
    def upgrade_state_dict_named(self, state_dict, name):
        

        for k in list(state_dict.keys()):
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

    def forward_embedding(self, src_tokens):
        padding_mask = src_tokens.eq(self.padding_idx)
        token_embedding = self.embed_tokens(src_tokens)
        x = embed = token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        # roberta layer norm
        x = self.emb_layer_norm(x)
        
        x = self.dropout_module(x)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        return x, embed, padding_mask

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
    ):
        x, encoder_embedding, encoder_padding_mask = self.forward_embedding(src_tokens)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        
        # encoder layers
        for layer in self.layers:
            x = layer.forward_encoder(x, encoder_padding_mask)
        
        return {
            "encoder_out": [x],
            "encoder_padding_mask": [encoder_padding_mask],
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
        
        self._left_attn_mask_collect = {}
        self._right_attn_mask_collect = {}

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
        step=0, 
        **unused
    ):
        features = self.extract_features(
            encoder_out=encoder_out,
            tgt_segment=tgt_segment,
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out

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

        return length_tgt

    def create_sparse_mask(self, context_padding_mask, span_padding_mask, scale=1.):
        
        encoder_dim = context_padding_mask.size(1)
        decoder_dim = span_padding_mask.size(1)
        window_size = int(32 * scale)

        if (
            window_size not in self._left_attn_mask_collect
            or encoder_dim > self._left_attn_mask_collect[window_size].size(1)
            or decoder_dim > self._left_attn_mask_collect[window_size].size(0)
        ):
            self._left_attn_mask_collect[window_size] = torch.zeros([decoder_dim, encoder_dim])
        else:
            self._left_attn_mask_collect[window_size] = self._left_attn_mask_collect[window_size][:decoder_dim, :encoder_dim]
        
        if (
            window_size not in self._right_attn_mask_collect 
            or decoder_dim > self._right_attn_mask_collect[window_size].size(0)
        ):
            self._right_attn_mask_collect[window_size] = torch.triu(utils.fill_with_neg_inf(torch.zeros([decoder_dim, decoder_dim])), window_size) + \
                                torch.tril(utils.fill_with_neg_inf(torch.zeros([decoder_dim, decoder_dim])), -window_size)
            self._right_attn_mask_collect[window_size][0, :] = 0
            self._right_attn_mask_collect[window_size][:, 0] = 0
        else:
            self._right_attn_mask_collect[window_size] = self._right_attn_mask_collect[window_size][:decoder_dim, :decoder_dim]

        attn_mask = torch.cat([self._left_attn_mask_collect[window_size], self._right_attn_mask_collect[window_size]], dim=1)
        return attn_mask.to(context_padding_mask.device)

    def extract_features(
        self,
        encoder_out=None,
        tgt_segment=None,
        **unused
    ):
        x, decoder_padding_mask = self.forward_embedding(tgt_segment)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        
        for i, layer in enumerate(self.layers):
            
            attn_scale = max((len(self.layers) - i) / len(self.layers) * 0.75, 0.125)
            attn_mask = self.create_sparse_mask(encoder_out["encoder_padding_mask"][0], decoder_padding_mask)

            x = layer.forward_decoder(
                x,
                encoder_out["encoder_out"][0],
                encoder_padding_mask=encoder_out["encoder_padding_mask"][0],
                decoder_padding_mask=decoder_padding_mask,
                attn_mask=attn_mask,
            )
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        return x
        
    def forward_embedding(self, src_tokens):
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


@register_model_architecture("plan_roberta", "plan_roberta")
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

@register_model_architecture("plan_roberta", "plan_roberta_large")
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
