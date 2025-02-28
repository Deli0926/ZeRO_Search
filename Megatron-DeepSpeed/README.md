Usage
=====
1. check `_03_summit.sh` file options in detail
```
_03_submit.sh # run sbatch shell
    ㄴ option: TP PP DP PARTITION ZERO_STAGE A_GPU_TYPE NUM_HETERO B_GPU_TYPE
```
2. run `_03_summit.sh` file

2-1. Homogeneous
example) 8 A10 nodes (TP, PP, DP, PARTITION, ZERO_STAGE) = (4, 2, 4, 20-20, 1)
   ```
   . _03_summit.sh 4 2 4 20-20 1 A10 1 - 
   ```
2-2. Heterogeneous
example) 5 A10 nodes, 3 A6000 nodes (TP, PP, DP, PARTITION, ZERO_STAGE) = (4, 2, 4, 20-20, 1)
  ```
  . _03_submit.sh 4 2 4 20-20 2 A10 1 A6000
  ```

in `_03_submit.sh` file, run configuration `_00_conf.sh` file and summit slurm jobs doing `sbatch _02_sbatch_EA10.sh`
