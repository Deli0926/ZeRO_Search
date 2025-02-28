#!/bin/bash

set -ex
hostname

TP=$1
PP=$2
DP=$3
PARTITION=$4
ZERO_STAGE=$5
SLURM_JOB_ID=$6
NODE_RANK=$7
MASTER_ADDR=$8
hostnode=$9

echo ======== RUN_INTER ========

echo TP: $TP
echo PP: $PP
echo DP: $DP
echo PARTITION: $PARTITION
echo ZERO_STAGE: $ZERO_STAGE
echo SLURM_JOB_ID: $SLURM_JOB_ID
echo NODE_RANK: $NODE_RANK
echo MASTER_ADDR: $MASTER_ADDR
echo hostnode: $hostnode

. _00_conf.sh $TP $PP $DP $PARTITION $ZERO_STAGE

BASE_PATH=/Megatron-DeepSpeed/dataset
DS_CONFIG=ds_config.json

PIPELINE_MODEL_PARALLEL_SPLIT_RANK=2 # Where the encoder ends within the pipeline group
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

L=40

if [ $MODEL == "llama2-13B" ]; then # LLAMA2 13B
        HIDDEN_SIZE=5120
        FFN_HIDDEN_SIZE=13824
        NUM_LAYERS=$L
        NUM_HEADS=40
        SEQ_LENGTH=4096
        NUM_KV_HEADS=8
        LR=3e-4
        MIN_LR=3e-5
        LR_WARMUP_STEPS=1
        WEIGHT_DECAY=0.1
        GRAD_CLIP=1
else
        echo error: invalid model argument MODEL only llama2-13B is allowed
        return 1
fi

MODEL_FAMILY=`echo "${MODEL%%-*}" | tr -d '0-9'`
echo "MODEL_FAMILY $MODEL_FAMILY"

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "overlap_comm": $OVERLAP
  },

  "fp16": {
    "enabled": true,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "weight_decay": "auto",
      "torch_adam": true,
      "adam_w_mode": true
    }
  }
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=${DS_CONFIG} ${ds_args}"
ds_args=" --zero-stage=${ZERO_STAGE} ${ds_args}"

if [ "$ZERO_STAGE" -ge 2]; then
    ds_args=" --no-pipeline-parallel ${ds_args}"
else
    ds_args=""
fi

DISTRIBUTED_ARGS="--nproc_per_node $NPROC_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

OUTPUT_ARGS="--log-interval 10 \
              --save-interval 100 \
              --eval-interval 100 \
              --eval-iters 10 $PROFILE_ARGS"

if [ $MODEL == "llama2-13B" ]; then
        TOKENIZER_PATH=${BASE_PATH}/tokenizer-13B.model # offical llama tokenizer.model
        DATA_PATH=${BASE_PATH}/llama2-13B-oscar_text_document
        VOCAB_FILE=${BASE_PATH}/gpt2-vocab.json
        MODEL_ARGS="--num-layers $NUM_LAYERS \
                --hidden-size $HIDDEN_SIZE \
                --ffn-hidden-size $FFN_HIDDEN_SIZE \
                --num-attention-heads $NUM_HEADS \
                --num-key-value-heads $NUM_KV_HEADS \
                --micro-batch-size $MICRO_BATCH_SIZE \
                --global-batch-size $GLOBAL_BATCH_SIZE \
                --seq-length $SEQ_LENGTH \
                --max-position-embeddings $SEQ_LENGTH \
                --train-iters 50 \
                --tokenizer-type GPTSentencePieceTokenizer \
                --tokenizer-model $TOKENIZER_PATH \
                --distributed-backend nccl \
                --lr $LR \
                --lr-decay-style cosine \
                --min-lr $MIN_LR \
                --weight-decay $WEIGHT_DECAY \
                --clip-grad $GRAD_CLIP \
                --lr-warmup-iters $LR_WARMUP_STEPS \
                --optimizer adam \
                --adam-beta1 0.9 \
                --adam-beta2 0.95 \
                --no-query-key-layer-scaling \
                --attention-dropout 0 \
                --hidden-dropout 0 \
                --use-rotary-position-embeddings \
                --untie-embeddings-and-output-weights \
                --swiglu \
                --normalization rmsnorm \
                --disable-bias-linear \
                --data-path $DATA_PATH --vocab-file $BASE_PATH/$VOCAB_FILE \
                --fp16 "
else
        echo "Model not supported"
        exit 1
fi

if [ $MODEL == "llama2-13B" ]; then
        RUN_TORCH_SCRIPT=$(cat << EOF
        torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
                $MODEL_ARGS \
                $OUTPUT_ARGS \
                --tensor-model-parallel-size $TP \
                --pipeline-model-parallel-size $PP \
                --balance $PARTITION \
                --no-async-tensor-model-parallel-allreduce \
                --no-gradient-accumulation-fusion \
                --split 100,0,0 $ds_args 
EOF
)
fi

function run_torch() {
        OMP_NUM_THREADS=4 $RUN_TORCH_SCRIPT
}

function run_torch_with_nsys() {
        NSYS_FILENAME=${SLURM_JOB_ID}_ds_z${ZERO_STAGE}_model${MODEL}_gb${GLOBAL_BATCH_SIZE}_mb${MICRO_BATCH_SIZE}_overlap_${OVERLAP}_TP${TP}_PP${PP}_DP${DP}
        OUTPUT_DIR=/Megatron-DeepSpeed/_log-nsys/$SLURM_JOB_ID
        mkdir -p $OUTPUT_DIR
        OMP_NUM_THREADS=4 nsys profile -t cuda,nvtx \
                --delay=5 \
                -o $OUTPUT_DIR/$NSYS_FILENAME-$NODE_RANK \
                --export=sqlite \
                -f true \
                $RUN_TORCH_SCRIPT
}

# if NSYS is true, then run nsys
if $NSYS; then
        echo "Run torch with nsys"
        run_torch_with_nsys
else
        echo "Run torch"
        gpustat -i > /Megatron-DeepSpeed/_log/$SLURM_JOB_ID/$hostnode.gpu &
        run_torch 
fi