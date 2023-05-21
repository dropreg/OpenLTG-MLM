#!/bin/bash
src=source
tgt=target

export CUDA_VISIBLE_DEVICES=0

data_dir=/opt/data/private/data/arggen/processed/data_bin
save_dir=/opt/data/private/ckpt/long2long_bert/arg_mask_long/

# --decoding-sampling \
python examples_acl/long2long_bert/src/generate.py $data_dir \
    --user-dir examples_acl/long2long_bert/src \
    --task plan_generation \
    --bpe gpt2 \
    --decoding-sampling \
    --gpt2-encoder-json examples_acl/long2long_bert/scripts/common/encoder.json \
    --gpt2-vocab-bpe examples_acl/long2long_bert/scripts/common/vocab.bpe \
    -s $src -t $tgt \
    --gen-subset test \
    --path $save_dir/checkpoint25.pt \
    --iter-decode-max-iter 8 \
    --iter-decode-with-beam 1 --remove-bpe \
    --iter-decode-force-max-iter \
    --batch-size 16 > $save_dir/inf_new.log

grep -a ^S $save_dir/inf_new.log | cut -f2 > $save_dir/test.src
grep -a ^T $save_dir/inf_new.log | cut -f2 > $save_dir/test.tgt
grep -a ^D $save_dir/inf_new.log | cut -f3 > $save_dir/test.hypo

# files2rouge $save_dir/test.tgt $save_dir/test.hypo

echo "generation down!"
