#!/bin/bash

# VOCAB_FILE=../dataset/gpt2-vocab.json
# MERGE_FILE=../dataset/gpt2-merges.txt

# IMPL=mmap
# python preprocess_data.py \
#        --input ../dataset/openwebtext_llama2_dataset-nopad.json \
#        --output-prefix llama2-13B-openwebtext \
#        --dataset-impl ${IMPL} \
#        --workers 8 \
#        --seq-length 4096 --tokenizer-type GPTSentencePieceTokenizer \
#        --tokenizer-model ../dataset/tokenizer-13B.model

export BASE_SRC_PATH="/Megatron-DeepSpeed"
export BASE_DATA_PATH="${BASE_SRC_PATH}/dataset"

python ${BASE_SRC_PATH}/tools/preprocess_data.py \
    --input ${BASE_DATA_PATH}/openwebtext_llama2_dataset-nopad.json \
    --output-prefix ${BASE_DATA_PATH}/llama2-13B-openwebtext \
    --vocab-file ${BASE_DATA_PATH}/gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPTSentencePieceTokenizer \
    --merge-file ${BASE_DATA_PATH}/gpt2-merges.txt --append-eod \
    --workers 8 \
    --seq-length 4096 \
    --tokenizer-model ${BASE_DATA_PATH}/tokenizer-13B.model