#!/bin/bash
src=source
tgt=target

export CUDA_VISIBLE_DEVICES=0,1,2,3

data_dir=/opt/data/private/data/arggen/processed/data_bin
save_dir=/opt/data/private/ckpt/long2long_bert/arg_mask_long/
roberta_path=/opt/data/private/data/roberta.base/model.pt


fairseq-train $data_dir \
    --reset-optimizer --reset-dataloader --reset-meters \
    --restore-file $roberta_path \
    --user-dir roberta/src \
    --task plan_generation \
    --arch plan_roberta \
    --share-all-embeddings \
    --dropout 0.1 --attention-dropout 0.1 \
    --lr 5e-5 \
    --criterion plan_loss \
    -s $src -t $tgt \
    --bpe gpt2 \
    --num-workers 0 \
    --gpt2-encoder-json roberta/scripts/common/encoder.json \
    --gpt2-vocab-bpe roberta/scripts/common/vocab.bpe \
    --truncate-source \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --max-tokens 4096 \
    --share-encoder-decoder \
    --max-perdict-length 512 \
    --no-progress-bar \
    --seed 64 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --find-unused-parameters \
    --lr-scheduler polynomial_decay --total-num-update 9000 --warmup-updates 500 \
    --max-epoch 25 \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 128 \
    --skip-invalid-size-inputs-valid-test \
    --save-dir $save_dir | tee -a $save_dir/train.log \
