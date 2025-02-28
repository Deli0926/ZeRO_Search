#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition={A6000 partition}
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=14
#SBATCH -o ./_log/%j.sbatch.%N.out         
#SBATCH -e ./_log/%j.sbatch.%N.err
#SBATCH --job-name=A6000
#************************************************************
GRES="gpu:a6000:4"
TP=$1
PP=$2
DP=$3
PARTITION=$4
CPUS_PER_TASK=14
GPU_TYPE=A6000
. _00_conf_hetero2.sh $TP $PP $DP $PARTITION
#************************************************************
mkdir -p ./_log2/$SLURM_JOB_ID

A6000_NODES=($A6000_NODES_STR)
function get_master_adress(){
    NODE_LIST=`scontrol show hostnames $SLURM_JOB_NODELIST`
    MASTER_HOST=`echo $NODE_LIST | awk '{print $1}'`
    MASTER_ADDR_LIST=`cat /etc/hosts | grep $MASTER_HOST | awk '{print $1}'`

    MASTER_ADDR_ARR=($MASTER_ADDR_LIST)
    MASTER_ADDR=${MASTER_ADDR_ARR[1]}
    echo A6000_NODES MASTER_ADDR_ARR:$MASTER_ADDR_ARR
    echo A6000_NODES MASTER_ADDR:$MASTER_ADDR
}

if [ "${A6000_NODES[0]}" = 0 ] ; then 
    get_master_adress
    read -ra MASTER_ADDR <<< "$MASTER_ADDR"
    echo ${MASTER_ADDR[0]} > MASTER_ADDR.txt
    echo ${MASTER_ADDR[0]}
    # save JOBID
    echo $SLURM_JOB_ID > MASTERJOBID.txt
else
    sleep 1
    MASTER_ADDR=$(<MASTER_ADDR.txt)
    # save JOBID
    echo $SLURM_JOB_ID >> SLAVEJOBID.txt

fi
echo A6000_NODES MASTER_ADDR:$MASTER_ADDR
echo A6000_NODES CONTAINER_PATH:$CONTAINER_PATH


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
                bash _01_run_inter_hetero2.sh $TP $PP $DP $PARTITION $SLURM_JOB_ID"

SRUN_SCRIPT=$(cat <<EOF

    $INIT_CONTAINER_SCRIPT

    free -m

    # hostname
    hostnode=\$(hostname -s) # save hostname
    echo "I am: \$hostnode"
    
    # SLURM_HOST_ARR
    SLURM_HOST_LIST=\$(scontrol show hostnames \$SLURM_JOB_NODELIST) # the array saving the number of slurm node in a row
    SLURM_HOST_ARR=(\$SLURM_HOST_LIST) # the array of slurm nodes
    echo "A6000 NODES $GPU_TYPE SLURM_HOST_ARR: \${SLURM_HOST_ARR[@]}" # the name of slurm node for A6000 nodes ex) [n097 n098 n099 n0100] 
    
    # A6000_NODES_INDEX_ARR
    A6000_NODES_INDEX_ARR=($A6000_NODES_STR) # defined in _00_conf_hetero2.sh
    echo "A6000 NODES $GPU_TYPE A6000_NODES_INDEX_ARR: \${A6000_NODES_INDEX_ARR[@]}" # the index of array for A6000 nodes ex) [1 3 6 7]

    for index in "\${!SLURM_HOST_ARR[@]}"; do
        h=\${SLURM_HOST_ARR[\$index]} # ex) h is n097

        if [[ "\$hostnode" == "\$h" ]]; then
            NODE_RANK=\${A6000_NODES_INDEX_ARR[\$index]} # NODE_RANK is 1
            echo "NODE_RANK : \$NODE_RANK"
            break 2
        fi
    done
    
    [ \$NODE_RANK == "0" ] && echo "I AM MASTER_NODE \$hostname" || echo ""    

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

