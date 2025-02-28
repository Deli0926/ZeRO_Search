"""
Portions of this code adapted from the 'AMP' project (https://github.com/DachengLi1/AMP).
@article{li2022amp,
  title={AMP: Automatically Finding Model Parallel Strategies with Heterogeneity Awareness},
  author={Li, Dacheng and Wang, Hongyi and Xing, Eric and Zhang, Hao},
  journal={arXiv preprint arXiv:2210.07297},
  year={2022}
}
"""

import time
import os
import argparse

import numpy as np
import pandas as pd
import torch

from utils import get_dp_method, no_placement_strategy_with_zero, get_node_map
from estimate import zerosearch
from device_placement import device_placement, get_cluster_list
from model_config import get_model_config

parser = argparse.ArgumentParser()
parser.add_argument("--node_type", type=str, nargs='*', help="the type of node type. for example, you have A10 A100 A6000")
parser.add_argument("--node_type_num_node", type=str, nargs="*", help="the number of each node. It will match with node_type.")
parser.add_argument("--num_node", type=int, default=8, help="the number of all nodes.")
parser.add_argument("--type", type=str, default="llama2_13B")
parser.add_argument("--gpu_per_node", type=int, default=4)
parser.add_argument("--custom_gbs", type=int, default=64)
parser.add_argument("--custom_num_layer", type=int, default=48)
parser.add_argument("--precision", type=int, default=16)
parser.add_argument("--iter", type=int, default=12_500_000, help="number of iterations for each experiment (default: 1)")
parser.add_argument("--pareto", action='store_true', help="True if you want to run pareto experiments (default: False)")
parser.add_argument("--add_exp_name", type=str, default="")
parser.add_argument("--exhaustive", action='store_true', help="True if you want to run exhaustive search for model partitioning (default: False)")
parser.add_argument("--comm_type", type=str, default="ib", help="eth / ib. communication protocol type between nodes")
parser.add_argument("--framework","-f", type=str, default="default", help="choose distributed deep learning framework [m: Megatron-LM / d: DeepSpeed / both: both] (default: both)")
parser.add_argument("--record", action='store_true', help="True if you want to save result as csv file using pandas DataFrame")
parser.add_argument("--pretty", action='store_true', help="True it is formed for report")
parser.add_argument("--search_space", action="store_true", help="True if you can get only search space give environment")
parser.add_argument("--search_method", type=str, default="minmax", help="It's meta-heuristic method, for example, you can select among ['minmax','ga']")
parser.add_argument("--profiledb_path", type=str, default="profiledb", help="Directory including profiledb")

args = parser.parse_args()
exhaustive_dict={}

if args.pareto:
    node_map = None
else:
    node_map = get_node_map(args)

if args.exhaustive:
    print("type parallelization strategy you want to search")
    mbs_exhaustive = int(1)
    tp_exhaustive = int(4)
    pp_exhaustive = int(4)
    dp_exhaustive = int(2)
    partition_exhaustive = [11, 10, 10, 11]
    exhaustive_dict = {"mbs":mbs_exhaustive, "tp":tp_exhaustive, "pp":pp_exhaustive,"dp":dp_exhaustive, "partition": partition_exhaustive}

gpu_per_node = args.gpu_per_node
num_node = args.num_node
node_type = args.node_type
node_type_num_node = args.node_type_num_node
record = args.record

home_path = os.environ['HOME']
pwd_path = os.environ['PWD']
dir_path = os.path.join(pwd_path,'logs')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
args.profiledb_path = os.path.join(os.pardir, args.profiledb_path)
    
record_file=f'N{args.num_node}_M{args.gpu_per_node}_model_{args.type}_comm_{args.comm_type}' + args.add_exp_name

if len(args.node_type) > 1:
    record_file += '_hetero'
else:
    record_file += '_homo'

cluster_info = {}

if args.pareto:
    A10 = [torch.tensor([100 * 1e9]).float(), torch.tensor([252 * 1e9]).float()]
    A100 = [torch.tensor([400 * 1e9]).float(), torch.tensor([1840* 1e9]).float()]
    A6000 = [torch.tensor([400 * 1e9]).float(), torch.tensor([1840* 1e9]).float()]
    
elif args.comm_type == "ib": # 4th generation UBAI cluster
    coff=0.6
    A10 = [torch.tensor([coff * 200 * 1e9]).float(), torch.tensor([coff * 32 * 8 * 1e9]).float()]
    A6000 = [torch.tensor([coff * 200 * 1e9]).float(), torch.tensor([coff * 32 * 8 * 1e9]).float()]
    RTX3090 = [torch.tensor([coff * 200 * 1e9]).float(), torch.tensor([coff * 32 * 8 * 1e9]).float()]
else: 
    assert args.comm_type not in ["ib"], "comm_type must be 'ib'"
    
dp_method = get_dp_method(args)
    
# get_cluster_list: the function which get each number of A10, A6000, RTX3090 node. depending on model type, it is different. default value is node num is 8
cluster_list = get_cluster_list(args, node_map)

# get model confing: can get GBS, hidden size, sequence length, num_layers, vocab_size, num_attention_heads, type(model name), precision depend on model
model_config, gbs = get_model_config(args)
want_simulate = []

time_s = time.time()
for cluster in cluster_list: # cluster_list is list of cluster combination. this for loop statement is for pareto options
    # cluster: ['0','0','1','1',...]
    
    # cheking cluster_list
    # assert False, f"cluster: {cluster}"
    if args.pareto:
        replace_dict = {str(i): node_type[i] for i in range(len(node_type))}
        node_map = {node_type[i]: 0 for i in range(len(node_type)) }
        for value in cluster.values():
            replaced_value = replace_dict[value]
            node_map[replaced_value] += 1
        cluster = list(cluster.values())
        num_node = sum(node_map.values())
        
    D = device_placement(node_map, cluster) # the whole array combination for heterogeneous/homogeneous nodes
    
    # checking permunation list
    # assert False, D
    # remove cache directory from last run
    if os.path.exists(os.path.join(home_path, "tmp")):
        for root, dirs, files in os.walk(os.path.join(home_path, "tmp")):
            for f in files:
                os.unlink(os.path.join(root, f))

    for d in D:
        s = time.time()
        # d is cyclic permutation
        # if node_type is A10 A6000, and you have one heterogeneous node, 
        # d is ['0', '0', '1', '1', ...], ['0', '1', '1', ..., '0'], ...
        # assert False, d
        print(f"d: {d}, node_map: {node_map}")
        node_placement = [ list(node_map.keys())[int(i)] for i in d ]
        print(node_placement)
        # node_placement is ['A10','A10','A100','A100',...]
        for i in range(len(node_placement)):
            if node_placement[i] == 'A10':
                d[i] = A10
            elif node_placement[i] == 'A100':
                d[i] = A100
            elif node_placement[i] == 'A6000':
                d[i] = A6000
            elif node_placement[i] == 'RTX3090':
                d[i] = RTX3090
            else:
                assert False, "Unknown node type"

        # assert False, f"{d}"
        model = zerosearch(args, model_config, exhaustive_dict, node_map)
        known = None

        # Estimating best configurations
        # num = 1
        while True:
            # print(f"i: {i}")
            # ret이 끝날때까지 무한 반복            
            ret = no_placement_strategy_with_zero(args, M=gpu_per_node, N=num_node, gbs=gbs, known=known, num_layers=model_config["num_layers"], dp_method=dp_method, exhaustive_dict=exhaustive_dict)
            
            if ret is None:
                break
            else:
                # i: zero stage , h: tp, w: pp, mbs: micro batch size, known: (i, tp, dp, pp) combination
                if args.exhaustive:
                    i, tp, dp, pp, mbs, known = ret
                    m = int(gbs / (dp * mbs))
                    parallel_dim = {"tp_deg": torch.ones(1,)*tp,  "pp_deg": torch.ones(1,)*pp, "dp_deg": torch.ones(1,)*dp}
                    # print(f"(gbs, mbs, tp, pp, dp, dp method): ({gbs}, {mbs}, {tp}, {pp}, {dp}, {i})")
                else:
                    i, tp, dp, mbs, known = ret
                    m = int(gbs / (dp * mbs))
                    pp = int((gpu_per_node*num_node)/(tp*dp))
                    parallel_dim = {"tp_deg": torch.ones(1,)*tp,  "pp_deg": torch.ones(1,)*pp, "dp_deg": torch.ones(1,)*dp}
                    
                fake_config = np.ones((gpu_per_node,num_node)) * (-1)
                model_args = (fake_config, gbs, mbs, d, model_config, parallel_dim, i) # model_args includes zero stage i 
                # print(f"model_args(fake_config, gbs, mbs, d, model_config, parallel_dim): {model_args}")
                # print(f"[{num}] model_args(mbs, parallel_dim): {mbs, i, parallel_dim}")
                
                # print(f"mbs, parallel_dim, zero-i: {mbs}, {parallel_dim}, {i}")
                # try:
                with torch.no_grad():
                    rank_map, partition, cost, pipecost, dp_side_cost, all_reduce_embedding_cost, \
                        is_oom, oom_gpumem = model(model_args, node_placement)
                # except:
                #     pass
                
                for k in parallel_dim:
                    parallel_dim[k] = int(parallel_dim[k].item())
                price_per_sec = 0.0
                for gpu in node_map:
                    # 0 A10 : g5.48xlarge, 1 A100, L4, H100 : p4d.24xlarge, g6.48xlarge, p5.48xlarge
                    # on demand price
                    price_table = {"A10": 1.4, "A100": 32.7726, "A6000": 16.200124, "L4": 13.3504, "H100": 98.32, "RTX3090":16.288 }
                    # price_table = {"A10": 2483, "A6000": 3025, "RTX3090": 2483 }
                    
                    try:
                        price_per_sec =  price_per_sec + ( price_table[gpu] / 3600) * node_map[gpu]
                        # print(f"price_table[gpu]: {price_table[gpu]}\nprice_per_sec: {price_per_sec}\nnode_map[gpu]: {node_map[gpu]}")
                    except:
                        pass

                # print(f"price_per_sec: {price_per_sec}")
                price_per_step = price_per_sec * cost.item() # price per second * second per step 
                pretrain_cost = price_per_step * args.iter
                # print(f"price_per_step: {price_per_step}\ncost: {cost.item()}")
                # TODO: add ZeRO, overlap_comm
                exp_partition =  [str(partition[0] - 1)] + [str(x) for x in partition[1:-1]] + [str(partition[-1] - 1)]
                exp_partition = "-".join(exp_partition)
                want_simulate.append((m, mbs, tp, pp, dp, i, False , node_placement, partition, cost.item(), pipecost.item(), dp_side_cost.item(), all_reduce_embedding_cost, is_oom, oom_gpumem.item(), pretrain_cost, price_per_step, exp_partition) + tuple(node_map.values()))
                # print(f"num: {num}")
                # num = num + 1                    
                
                    
        e = time.time()
        # print(f"node placement search time: {e - s:0.5f}")
print(f"Finished {time.time() - time_s:.5f}")

# sorted_settings = sorted(want_simulate, key = lambda kv: kv[8])
columns = ['m','mbs','tp','pp','dp','dp method', 'overlap_comm','node placement', 'partition', \
                                              'estimated time (s/step)','pipeline time','DP all-reduce time','Emb layer all-reduce time', \
                                              'is_oom','gpumem','train_cost','price_per_step','exp_partition'] + list(node_map.keys())

# 2024.03.19 add columns name as ZeRO, overlap_comm. delete zerooom_gpumem, is_zerooom
df = pd.DataFrame(want_simulate, columns = columns)

# rank column: ranking steptime
df['rank'] = df['estimated time (s/step)'].rank(method='min', ascending=True)

# real rank column: without oom, ranking steptime
df['real_rank'] = float('nan')
real_rank_df = df[df['is_oom'] == False]
real_rank_df['real_rank'] = real_rank_df['estimated time (s/step)'].rank(method='min')

# Convert ranks to integer
# real_rank_df['real_rank'] = real_rank_df['real_rank'].astype(int)


# insert real rank column to original df DataFrame
df.loc[real_rank_df.index, 'real_rank'] = real_rank_df['real_rank']

if args.pretty:
# arrange column name
    pretty_columns = ['rank','real_rank','m', 'mbs','tp','pp','dp','dp method'] + list(node_map.keys()) + ['estimated time (s/step)','pipeline time','DP all-reduce time','Emb layer all-reduce time', \
                                              'gpumem','is_oom','node placement', 'exp_partition', 'partition','train_cost','price_per_step']
    df = df[pretty_columns]
# else:
#     df = df[['rank','real_rank','m', 'mbs','tp','pp','dp','dp method', 'overlap_comm','node placement', 'partition', \
#                                                 'estimated time (s/step)','pipeline time','DP all-reduce time','Emb layer all-reduce time', \
#                                                 'is_oom','gpumem','train_cost','price_per_step']]
df['rank'] = df['rank'].astype(int)
df['real_rank'] = df['real_rank'].fillna(0).astype(int)

df = df.sort_values(by='estimated time (s/step)')

# first remove existing csv file

if record:
    if os.path.exists(f"{os.path.join(dir_path, record_file)}.csv"):
        os.remove(f"{os.path.join(dir_path, record_file)}.csv")
    df.to_csv(f"{os.path.join(dir_path, record_file)}.csv", index=False)

    print("csv file saved at: ", f"{os.path.join(dir_path, record_file)}.csv")
else:
    print("Done!")
