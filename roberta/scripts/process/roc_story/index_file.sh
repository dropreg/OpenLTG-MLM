# RAW_DATA=/opt/data/private/data/arr1015_data/roc_story
# PROCESSED_DATA=/opt/data/private/data/arr1015_data/roc_story/processed/

RAW_DATA=/opt/data/private/data/arr1015_data/roc_story/seg_two
PROCESSED_DATA=/opt/data/private/data/arr1015_data/roc_story/seg_two/processed/

mkdir -p $PROCESSED_DATA

for PREFIX in {train,val,test}
do 

    echo $PREFIX

    python examples/long2long_bert/scripts/common/processing_bpe_encoder.py \
        --encoder-json examples/long2long_bert/scripts/common/encoder.json \
        --vocab-bpe examples/long2long_bert/scripts/common/vocab.bpe \
        --inputs "$RAW_DATA/$PREFIX.sent.source" \
        --outputs "$PROCESSED_DATA/$PREFIX.split.source" \
        --workers 40 \
        --keep-empty;

    python examples/long2long_bert/scripts/common/processing_bpe_encoder.py \
        --encoder-json examples/long2long_bert/scripts/common/encoder.json \
        --vocab-bpe examples/long2long_bert/scripts/common/vocab.bpe \
        --inputs "$RAW_DATA/$PREFIX.sent.target" \
        --outputs "$PROCESSED_DATA/$PREFIX.split.target" \
        --workers 40 \
        --keep-empty;

done