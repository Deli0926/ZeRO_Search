#!/bin/bash

CONTAINER_IMAGE_PATH="../container/deepspeed.2406.sqsh"
CONTAINER_NAME="deepspeed2406"
CONTAINER_PATH=/enroot/$UID/data/$CONTAINER_NAME

TP=$1
PP=$2
DP=$3
PARTITION=$4
ZERO_STAGE=$5

MODEL="llama2-13B"
NNODES=8
NPROC_PER_NODE=4

MICRO_BATCH_SIZE=1
NUM_MB=1
# GLOBAL_BATCH_SIZE=64
GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIZE * NUM_MB * DP))
# L: llama2-13B:40

NSYS=false
PROFILE=false

MASTER_PORT=7777
RELOAD_CONTAINER=false

OVERLAP=false

PIPELINE_MODEL_PARALLEL_SPLIT_RANK=2 # Where the encoder ends within the pipeline group
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "MBS: $MICRO_BATCH_SIZE"
echo "TP: $TP"
echo "PP: $PP"
echo "DP: $DP"
echo "PARTITION: $PARTITION"
echo ZERO: $ZERO_STAGE

# set model specific arguments
echo "MODEL : $MODEL"
if [ $MODEL == "llama2-13B" ]; then # LLAMA2 13B
        HIDDEN_SIZE=5120
        FFN_HIDDEN_SIZE=13824
        NUM_LAYERS=40
        NUM_HEADS=40
        SEQ_LENGTH=4096
        NUM_KV_HEADS=8
        LR=3e-4
        MIN_LR=3e-5
        LR_WARMUP_STEPS=1
        WEIGHT_DECAY=0.1
        GRAD_CLIP=1
else
        echo error: invalid model argument MODEL only "llama2-13B" is allowed
        return 1
fi

# set zero arguments
if [ "$ZERO_STAGE" == "0" ]; then
        echo "ZERO OFF"
else
        echo "ZERO STAGE: $ZERO_STAGE"
fi

# set overlap arguments
if $OVERLAP; then
        echo "OVERLAP: $OVERLAP"
else
        echo "OVERLAP OFF"
fi

# set profiling arguments
if $PROFILE; then
        echo "Profiling"
        PROFILE_ARGS="--timing-log-level 2 --timing-log-option all"
else
        echo "Not profiling"
        PROFILE_ARGS=""
fi

# set nsight arguments
if $NSYS; then
        echo "NSYS: $NSYS"
else
        echo "NSYS OFF"
fi