

# PROCESSED_DATA=/opt/data/private/data/arr1015_data/writing_prompt_min10/processed/
# PROCESSED_DATA=/opt/data/private/data/arr1015_data/writing_prompt/seg_two/processed/
# PROCESSED_DATA=/opt/data/private/data/arr1015_data/writing_prompt/seg_four/processed/
PROCESSED_DATA=/opt/data/private/data/arr1015_data/writing_prompt/seg_eight/processed/

fairseq-preprocess \
    --user-dir examples/long2long_bert/src \
    --task open_generation \
    --source-lang source --target-lang target \
    --trainpref "$PROCESSED_DATA/train" \
    --validpref "$PROCESSED_DATA/valid" \
    --testpref "$PROCESSED_DATA/test" \
    --destdir "$PROCESSED_DATA/data_bin" \
    --workers 60 \
    --srcdict /opt/data/private/data/roberta.base/dict.txt \
    --tgtdict /opt/data/private/data/roberta.base/dict.txt \
