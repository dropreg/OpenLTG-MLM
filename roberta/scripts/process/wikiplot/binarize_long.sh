

RAW_DATA=/opt/data/private/data/acl2023_data/wikiplot
PROCESSED_DATA=/opt/data/private/data/acl2023_data/wikiplot/processed/

mkdir -p $PROCESSED_DATA

for PREFIX in {train,valid,test}
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
    --user-dir roberta/src \
    --task open_generation \
    --source-lang source --target-lang target \
    --trainpref "$PROCESSED_DATA/train" \
    --validpref "$PROCESSED_DATA/valid" \
    --testpref "$PROCESSED_DATA/test" \
    --destdir "$PROCESSED_DATA/data_bin" \
    --workers 60 \
    --srcdict /opt/data/private/data/roberta.base/dict.txt \
    --tgtdict /opt/data/private/data/roberta.base/dict.txt \
