#!/bin/bash
src=source
tgt=target

export CUDA_VISIBLE_DEVICES=0,1,2,3

data_dir=/opt/data/private/data/arr1015_data/roc_story/long/data_bin
save_dir=/opt/data/private/ckpt/long2long_bert/roc_raw_long/
roberta_path=/opt/data/private/data/roberta.base/model.pt


fairseq-train $data_dir \
    --reset-optimizer --reset-dataloader --reset-meters \
    --restore-file $roberta_path \
    --user-dir roberta/src \
    --task open_generation \
    --arch long_roberta \
    --share-all-embeddings \
    --dropout 0.1 --attention-dropout 0.1 \
    --lr 5e-5 \
    --criterion span_label_smooth_loss \
    -s $src -t $tgt \
    --truncate-source \
    --max-tokens 4096 \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --share-encoder-decoder \
    --max-perdict-length 128 \
    --no-progress-bar \
    --seed 64 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --find-unused-parameters \
    --lr-scheduler polynomial_decay --total-num-update 5000 --warmup-updates 500 \
    --max-epoch 15 \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 128 \
    --skip-invalid-size-inputs-valid-test \
    --save-dir $save_dir | tee -a $save_dir/train.log \
