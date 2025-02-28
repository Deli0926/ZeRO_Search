ZeROSearch
========
#KJH33
# overview
Large language models cannot fit within a single GPU memory, making 3D parallel deep learning across multiple GPUs essential. ZeRO technique, on the other hand, can further reduce the memory footprint of optimizer states, gradients, and model parameters per GPU and can also be combined with Pipeline Parallelism. Depending on the model size and the cluster architecture and configuration, such as the number of GPUs and their communication bandwidth, 3D parallel deep learning without ZeRO or with ZeRO may outperform the other. The search space is significantly large, consisting of the type and the number of GPUs, the communication bandwidth among them, the 3D parallel degrees, the model partitioning when the Pipeline Parallelism degree is used, and the level of ZeRO to employ. It becomes even larger when we consider a heterogeneous cluster where more than one GPU type is used. This paper analyzes the execution of different levels of ZeRO and accurately estimates the execution latency. It presents a fast design space search framework called ZeROSearch, which can automatically find the optimal design point in the search space. Compared to the 3D parallel DL without ZeRO, we explore 24 times larger search space, and ZeROSearch finds that 3D parallel DL with ZeRO Stage 1 leads to the minimal training time in the heterogeneous cluster with 128 GPUs with Llama2 13B.

# setup
1. make conda virtual environment
   ```
   conda create -n py310 python=3.10
   ```
2. install packages
   ```
   conda activate py310
   pip install torch numpy pandas
   ```
# directory hierarchy
```
ZeROSearch
ㄴprofiledb
ㄴsrc
    ㄴzerosearch.py
    ㄴestimate.py
    ㄴpipe.py
    ㄴstage.py
    ㄴutils.py
    ㄴdevice_placement.py
    ㄴmodel_config.py
```
## profiledb
includes profile DB which is the execution time for each layer type of the given large language model. The layer types are embedding layer, transformer layer, post process layer.

# Usage

1. Create virtual environment
2. Install packages
3. Execute zerosearch framework with arguments
4. (If you save logs) Check the result of predictions.

```
python zerosearch.py {arg1, arg2, ...}
```

## argument
- `node_type`: The type of nodes. For example, you can have A10, A100, A6000. the type of `node_type` is string list.
- `num_node`: The number of nodes. If homogeneous, you have to use it.
- `type`: The type of LLM model. you can have only Llama2 13B. If you want to add another LLM, should create profile DB.
- `comm_type`: The type of communication between nodes. you can have only ib which is infiniband card.
- `record`: If `True`, save the result of prediction.
- `pretty`: If `True`, align columns of result. the columns are `rank` `real_rank` `m` `mbs` `tp` `pp` `dp` `dp_method` `A10`(`A100`, `A6000`) `estimated time(s/step)` `pipeline time` `DP all-reduce time` `Emb layer all-reduce time` `gpumem` `is_oom` `node placement` `exp_partition` `partition` `train_cost` `price_per_step`
    - `rank`: The rank of combinations. `real_rank` is the rank except `is_oom` True.
    - `dp_method`: The method of Data Parallelism. It includes `zero0` `zero1` `zero2` `zero3`.
- `pareto`: If `True`, you run pareto experiments.
- `exhaustive`: If `True`, you can get predictions of the specific configuration `mbs` `tp` `pp` `dp` `partition`.
- `search_method`: The meta-heuristic search methohd. you can select among `minmax` `ga`. `ga` is Genetic Algorithms.
  
## example

The main python file of ZeroSearch framework is in the `src` directory. If you use it, you enter `src` directory.
```
cd src
```

- homegeneous 8 A10 nodes with Llama2 13B

```
python zerosearch.py --node_type A10 --num_node 8 --type llama2_13B --framework d
```
- heterogeneous 3 A10, 5 A6000 nodes with Llama2 13B
```
python zerosearch.py --node_type A10 A6000 --node_type_num_node 3 5 --type llama2_13B --framework d
```
- get pareto graph. heterogeneous 3 A10, 5 A6000 nodes with Llama2 13B
```
python zerosearch.py --node_type A10 A6000 --node_type_num_node 3 5 --type llama2_13B --framework d
```

