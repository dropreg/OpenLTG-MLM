


# RAW_DATA=/opt/data/private/data/arr1015_data/writing_prompt_min10
# PROCESSED_DATA=/opt/data/private/data/arr1015_data/writing_prompt_min10/processed/

# RAW_DATA=/opt/data/private/data/arr1015_data/writing_prompt/seg_two/
# PROCESSED_DATA=/opt/data/private/data/arr1015_data/writing_prompt/seg_two/processed/

# RAW_DATA=/opt/data/private/data/arr1015_data/writing_prompt/seg_four/
# PROCESSED_DATA=/opt/data/private/data/arr1015_data/writing_prompt/seg_four/processed/

RAW_DATA=/opt/data/private/data/arr1015_data/writing_prompt/seg_eight/
PROCESSED_DATA=/opt/data/private/data/arr1015_data/writing_prompt/seg_eight/processed/

mkdir -p $PROCESSED_DATA

for PREFIX in {train,valid,test}
do 

    echo $PREFIX

    python examples/long2long_bert/scripts/common/processing_bpe_encoder.py \
        --encoder-json examples/long2long_bert/scripts/common/encoder.json \
        --vocab-bpe examples/long2long_bert/scripts/common/vocab.bpe \
        --inputs "$RAW_DATA/$PREFIX.source" \
        --outputs "$PROCESSED_DATA/$PREFIX.split.source" \
        --workers 40 \
        --keep-empty;

    python examples/long2long_bert/scripts/common/processing_bpe_encoder.py \
        --encoder-json examples/long2long_bert/scripts/common/encoder.json \
        --vocab-bpe examples/long2long_bert/scripts/common/vocab.bpe \
        --inputs "$RAW_DATA/$PREFIX.target" \
        --outputs "$PROCESSED_DATA/$PREFIX.split.target" \
        --workers 40 \
        --keep-empty;

done