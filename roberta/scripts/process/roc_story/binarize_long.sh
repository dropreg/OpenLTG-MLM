

RAW_DATA=/opt/data/private/data/arr1015_data/roc_story
PROCESSED_DATA=/opt/data/private/data/arr1015_data/roc_story/long/

for PREFIX in {train,val,test}
do 

    echo $PREFIX
    python roberta/scripts/common/processing_bpe_encoder.py \
        --encoder-json roberta/scripts/common/encoder.json \
        --vocab-bpe roberta/scripts/common/vocab.bpe \
        --inputs "$RAW_DATA/$PREFIX.source" \
        --outputs "$PROCESSED_DATA/$PREFIX.source" \
        --workers 40 \
        --keep-empty;
    
    python roberta/scripts/common/processing_bpe_encoder.py \
        --encoder-json roberta/scripts/common/encoder.json \
        --vocab-bpe roberta/scripts/common/vocab.bpe \
        --inputs "$RAW_DATA/$PREFIX.target" \
        --outputs "$PROCESSED_DATA/$PREFIX.target" \
        --workers 40 \
        --keep-empty;

done

fairseq-preprocess \
    --source-lang source --target-lang target \
    --trainpref "$PROCESSED_DATA/train" \
    --validpref "$PROCESSED_DATA/val" \
    --testpref "$PROCESSED_DATA/test" \
    --destdir "$PROCESSED_DATA/data_bin" \
    --workers 60 \
    --srcdict /opt/data/private/data/roberta.base/dict.txt \
    --tgtdict /opt/data/private/data/roberta.base/dict.txt \

