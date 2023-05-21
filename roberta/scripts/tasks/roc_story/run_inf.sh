#!/bin/bash
src=source
tgt=target

export CUDA_VISIBLE_DEVICES=0

data_dir=/opt/data/private/data/arr1015_data/roc_story/long/data_bin
save_dir=/opt/data/private/ckpt/long2long_bert/roc_long/


python roberta/src/generate.py $data_dir \
    --user-dir roberta/src \
    --task open_generation \
    --span-sampling \
    --bpe gpt2 \
    --gpt2-encoder-json roberta/scripts/common/encoder.json \
    --gpt2-vocab-bpe roberta/scripts/common/vocab.bpe \
    -s $src -t $tgt \
    --seed 64 \
    --gen-subset test \
    --path $save_dir/checkpoint30.pt \
    --iter-decode-max-iter 1 \
    --iter-decode-with-beam 1 --remove-bpe \
    --iter-decode-force-max-iter \
    --batch-size 32 > $save_dir/inf_new.log

grep -a ^S $save_dir/inf_new.log | cut -f2 > $save_dir/test.src
grep -a ^T $save_dir/inf_new.log | cut -f2 > $save_dir/test.tgt
grep -a ^D $save_dir/inf_new.log | cut -f3 > $save_dir/test.hypo

echo "generation down!"
