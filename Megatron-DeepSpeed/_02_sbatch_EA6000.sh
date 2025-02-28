#!/bin/bash
#SBATCH --nodes=8
#SBATCH --partition={partition name}
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=14
#SBATCH -o ./_log/%j.sbatch.%N.out         
#SBATCH -e ./_log/%j.sbatch.%N.err
#SBATCH -J LLAMA2

#************************************************************
GRES="gpu:a6000:4"
TP=$1
PP=$2
DP=$3
PARTITION=$4
ZERO_STAGE=$5
CPUS_PER_TASK=14
GPU_TYPE=A6000
. _00_conf.sh $TP $PP $DP $PARTITION $ZERO_STAGE
#************************************************************

SHELL_PATH=`pwd -P`
echo $SHELL_PATH

mkdir -p ./_log/$SLURM_JOB_ID # log running SBATCH shell

function get_master_adress(){
    NODE_LIST=`scontrol show hostnames $SLURM_JOB_NODELIST`
    MASTER_HOST=`echo $NODE_LIST | awk '{print $1}'`
    MASTER_ADDR_LIST=`cat /etc/hosts | grep $MASTER_HOST | awk '{print $1}'`
    MASTER_ADDR_ARR=($MASTER_ADDR_LIST)
    MASTER_ADDR=${MASTER_ADDR_ARR[1]}
}
get_master_adress

echo MASTER_ADDR:$MASTER_ADDR
echo CONTAINER_PATH:$CONTAINER_PATH
echo CONTAINER_NAME:$CONTAINER_NAME
echo CONTAINER_IMAGE_PATH:$CONTAINER_IMAGE_PATH

INIT_CONTAINER_SCRIPT=$(cat <<EOF
    
    if $RELOAD_CONTAINER ; then
        rm -rf $CONTAINER_PATH
    fi

    if [ -d "$CONTAINER_PATH" ] ; then 
        echo "container exist";
    else
        enroot create -n $CONTAINER_NAME $CONTAINER_IMAGE_PATH ;
    fi

EOF
)

ENROOT_SCRIPT="cd /Megatron-DeepSpeed && \
                bash _01_run_inter_homo.sh $TP $PP $DP $PARTITION $ZERO_STAGE $SLURM_JOB_ID"

SRUN_SCRIPT=$(cat <<EOF

    $INIT_CONTAINER_SCRIPT

    NODE_LIST=\`scontrol show hostnames \$SLURM_JOB_NODELIST\`
    node_array=(\$NODE_LIST)
    length=\${#node_array[@]}
    hostnode=\`hostname -s\`
    for (( index = 0; index < length ; index++ )); do
        node=\${node_array[\$index]}
        if [ \$node == \$hostnode ]; then
            NODE_RANK=\$index
        fi
    done 

    enroot start --root \
                --rw \
                -m $HOME/ZeROSearch/Megatron-DeepSpeed:/Megatron-DeepSpeed \
                $CONTAINER_NAME \
                bash -c "$ENROOT_SCRIPT \$NODE_RANK $MASTER_ADDR \$hostnode"
EOF
)

srun --partition=$SLURM_JOB_PARTITION \
      --gres=$GRES \
      --cpus-per-task=$CPUS_PER_TASK \
      -o ./_log/%j/%N.out \
      -e ./_log/%j/%N.err \
      bash -c "$SRUN_SCRIPT"

