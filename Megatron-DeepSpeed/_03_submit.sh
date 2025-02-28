#!/bin/bash
TP=$1
PP=$2
DP=$3
PARTITION=$4
ZERO_STAGE=$5
# NUM_HETERO == 1 GPU TYPE, NUM_HETERO == 2 A GPU TYPE
A_GPU_TYPE=$6 # A10, A6000
NUM_HETERO=$7 # the number of heterogeneous nodes
B_GPU_TYPE=A6000

if  [ $NUM_HETERO == "1" ]; then
    . _00_conf.sh $TP $PP $DP $PARTITION $ZERO_STAGE
    sbatch_job="_02_sbatch_E$A_GPU_TYPE.sh"
    echo $sbatch_job
    sbatch $sbatch_job $TP $PP $DP $PARTITION $ZERO_STAGE
    echo "$A_GPU_TYPE homogeneous"

elif [ $NUM_HETERO == "2" ]; then
    . _00_conf_hetero2.sh $TP $PP $DP $PARTITION
    sbatch _02_sbatch_hetero2_$A_GPU_TYPE.sh $TP $PP $DP $PARTITION
    sbatch _02_sbatch_hetero2_$B_GPU_TYPE.sh $TP $PP $DP $PARTITION
    
elif [ $NUM_HETERO == "3" ]; then
    . _00_conf.sh $TP $PP $DP $PARTITION
    ARUN_FILE="ARUN.txt"
    BRUN_FILE="BRUN.txt"
    CRUN_FILE="CRUN.txt"

    if [ -e $ARUN_FILE ]; then 
        rm $ARUN_FILE
        echo "remove $ARUN_FILE" 
    fi
    
    if [ -e $BRUN_FILE ]; then
        rm $BRUN_FILE 
        echo "remove $BRUN_FILE"
    fi

    if [ -e $CRUN_FILE ]; then
        rm $CRUN_FILE 
        echo "remove $CRUN_FILE"
    fi

    echo "false" > $ARUN_FILE
    echo "false" > $BRUN_FILE
    echo "false" > $CRUN_FILE

    echo ${A_NODES[@]}
    echo ${B_NODES[@]}
    echo ${C_NODES[@]}

    sbatch _02_sbatch_A.sh $TP $PP $DP $PARTITION
    sbatch _02_sbatch_B.sh $TP $PP $DP $PARTITION
    sbatch _02_sbatch_C.sh $TP $PP $DP $PARTITION
    
    # waiting if no nodes
    if $REAPEAT ; then
        TIMES=3600 # unit: s
        #HOUR * TIMES = wait time
        HOUR=1
        SLEEP_TIME=$((TIMES * HOUR))
        echo "SLEEP_TIME(s): $SLEEP_TIME"
        echo "now repeat"
        while [ true ]
                do
                    # check the status of nodes
                    sleep 100

                    NOW=$(date +"%Y%m%d %H-%M-%S")
                    echo "waiting: $NOW"
                    # write each condition of node
                    ARUN=$(<ARUN.txt)
                    BRUN=$(<BRUN.txt)
                    CRUN=$(<CRUN.txt)
                    
                    if  ( [ "$ARUN" = "true" ] && [ "$BRUN" = "true" ] && [ "$CRUN" = "true" ] )  ; then
                        NOW=$(date +"%Y%m%d %H-%M-%S")
                        echo "waiting: $NOW"
                        echo "waiting to $SLEEP_TIME hours"

                        ARUN_FILE="ARUN.txt"
                        BRUN_FILE="BRUN.txt"
                        CRUN_FILE="CRUN.txt"

                        if [ -e $ARUN_FILE ]; then 
                            rm $ARUN_FILE 
                            echo "remove $ARUN_FILE" 
                        fi
                        
                        if [ -e $BRUN_FILE ]; then
                            rm $BRUN_FILE 
                            echo "remove $BRUN_FILE"
                        fi

                        if [ -e $CRUN_FILE ]; then
                            rm $CRUN_FILE 
                            echo "remove $CRUN_FILE"
                        fi
                    
                        echo "false" > $ARUN_FILE
                        echo "false" > $BRUN_FILE
                        echo "false" > $CRUN_FILE
                        
                        sleep $SLEEP_TIME
                        # save the result
                        # echo "result saving"
                        # . result.sh
                        # sleep 15
                        # run sbatch
                        sbatch _02_sbatch_A.sh $TP $PP $DP $PARTITION
                        sbatch _02_sbatch_B.sh $TP $PP $DP $PARTITION
                        sbatch _02_sbatch_C.sh $TP $PP $DP $PARTITION
                        
                    fi
                    
                done
    fi
fi


