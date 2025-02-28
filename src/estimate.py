"""
Portions of this code adapted from the 'AMP' project (https://github.com/DachengLi1/AMP). 
@article{li2022amp,
  title={AMP: Automatically Finding Model Parallel Strategies with Heterogeneity Awareness},
  author={Li, Dacheng and Wang, Hongyi and Xing, Eric and Zhang, Hao},
  journal={arXiv preprint arXiv:2210.07297},
  year={2022}
}
"""
from argparse import Namespace
from collections import defaultdict

import os

import torch
import torch.nn as nn
import numpy as np

from utils import axis2rank
from pipe import minmax, schedule, get_stage_latency, explain_minmax, exhaustive_partition, ga
from device_placement import get_gpu_for_stage

import copy

home_dir = os.environ['HOME'] 

class zerosearch(nn.Module):
    def __init__(self, args:Namespace, model_config, exhaustive_dict, node_map):
        
        super().__init__()
        self.args = args
        self.model_config = model_config
        self.model_type = model_config["type"]
        self.num_node = args.num_node
        self.comm_type = args.comm_type
        self.exhaustive_dict = exhaustive_dict
        self.node_map = node_map
        self.profiledb_path = args.profiledb_path
        self.tp_array = [1, 2, 4]
        if args.gpu_per_node > 4:
            self.tp_array.append(8)
        
    def init_param(self, dp_method, node_map):
        # zerosearch_config directory stores the real forward time with correponding tensor parallel degree.
        zerosearch_config={}  
        gpu_type = node_map.keys()
        
        if dp_method in ["zero0", "zero1", "zero2", "zero3"]:
            add_name = "_ds"
        
        for g in gpu_type:
            profile_db={}
            for mp_size in self.tp_array:
                times = 1.0
                if dp_method == "zero3":
                    times = times * 1.6
                    known_record = f"{self.profiledb_path}/{self.model_type}_{g}_{mp_size}{add_name}"
                if dp_method in ["zero0","zero1","zero2","zero3"]:
                    times = times * 4 # backward / forward = 3, forward + backward = 4 forward
                    known_record = f"{self.profiledb_path}/{self.model_type}_{g}_{mp_size}{add_name}"
                else:
                    times = times * 3 # backward / forward = 2, forward + backward = 3 forward
                    known_record = f"{self.profiledb_path}/{self.model_type}_{g}_{mp_size}"
                # print(f"profiledb path: {known_record}")
                profile_cost = times * np.load(f"{known_record}.npy") 
                profile_db[str(mp_size)]=profile_cost
            zerosearch_config[g]=profile_db
        return zerosearch_config
        
    def forward(self, model_args, node_placement):
        config, bs, micro_bs, cluster_info, model_config, oth, i = model_args
        zerosearch_config = self.init_param(i, self.node_map)
        # print(f"i, zerosearch_config: {i} {zerosearch_config}")
        # assert False, "Done!"
        rank_map, partition, zerosearch_pred, pipecost, dp_side_cost, all_reduce_embedding_cost, \
            is_oom, gpumem = predict(self.args, config, bs, micro_bs, cluster_info, model_config, zerosearch_config, oth, node_placement, i, self.exhaustive_dict, self.node_map)
        return rank_map, partition, zerosearch_pred, pipecost, dp_side_cost, all_reduce_embedding_cost, is_oom, gpumem
        
# pipeline communication cost, return shape: (L-1, pp-1)
def get_cost_c(args:Namespace, cluster_info, model_config, parallel_config, _layer=None):
    h = model_config["hidden_size"]
    s = model_config["sequence_length"]
    n = model_config["num_layers"]
    bs = parallel_config["micro_bs"]
    rank_node_map = parallel_config["rank_node_map"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]
    comm_type=args.comm_type
    
    # print(f"tp: {int(mp.item())}, dp: {int(dp.item())}, pp: {int(pp.item())}, comm_type: {comm_type}")
    
    precision = torch.ones(1)*16 # TODO: support fp32, should be args.precision

    _num_layer = len(_layer)

    if pp == 1:
        # print(f"pp: {pp}")
        return torch.zeros(int(n.item())), _layer
      
    # build layer activation lookup table.
    layer_volume = []
    last_volume = torch.zeros(1,)
    for i in range(_num_layer):
        layer_type = _layer[i]
        if layer_type == "embedding_layer" or layer_type == "transformer_layer" or layer_type == "encoder" or layer_type == "decoder" or layer_type == "post_process":
            last_volume = bs * s * h
            layer_volume.append(last_volume)
        else:
            # unrecognized layer type
            raise ValueError(f"Unrecognized layer type: {layer_type}")
    # Build communication cost between pipeline stages by looking up the cluster information
    cost_c = torch.zeros((int(dp.item()), _num_layer, int(pp.item()-1)))
    for i in range(int(dp.item())):    
        for j in range(int(pp.item()-1)):
            # get the slowest mp gpu connection
            slowest_bandwidth = np.inf
            for k in range(int(mp.item())):    
                rank_cur = axis2rank(axis=(j,i,k), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                rank_peer = axis2rank(axis=(j+1,i,k), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                node_cur = rank_node_map[int(rank_cur.item())]
                node_peer = rank_node_map[int(rank_peer.item())]
                
                if node_cur != node_peer: 
                    cur_bandwidth = min(cluster_info[node_cur][0], cluster_info[node_peer][0])
                    cur_bandwidth = cur_bandwidth / int(dp.item()) / int(mp.item()) / 1.05
                else:
                    cur_bandwidth = cluster_info[node_cur][1] / 3.5
                            
                if cur_bandwidth < slowest_bandwidth:
                    slowest_bandwidth = cur_bandwidth     
                    
            for k in range(_num_layer):
                cost_c[i][k][j] = layer_volume[k] * precision / slowest_bandwidth
    cost_c = torch.max(cost_c, dim=0)
    return cost_c.values, _layer

# execution cost for one layer, return shape (L,)
def get_cost_e(model_config, parallel_config, profile_cost, _layer, model_type):    
    n = model_config["num_layers"]
    bs = parallel_config["micro_bs"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]
    _num_layer = len(_layer)
    
    # print(f" _num_layer: {_num_layer}")
    
    cost_e = dict.fromkeys(profile_cost)
    # print(f"cost_e: {cost_e}")
    for g in profile_cost:
        cost_e_arr = np.zeros((int(dp.item()), _num_layer))
        for i in range(int(dp.item())):
            assert _num_layer == len(profile_cost[g]['1']), f"predicted number of layers not equal to actual. {profile_cost[g]['1']}"
            # cost_e in the main result is equivalent to using profile_cost.
            embedding_layer_counted = False
            for layer_id in range(_num_layer):
                layer_type = _layer[layer_id]
                #################################################################
                #                      GPT & LLAMA2
                #################################################################
                if model_type in ["gpt2XL","llama2_13B","llama2_13B_mini"]:
                    if layer_type == "embedding_layer" or layer_type == "post_process":
                        if g == "A100": # A100 GPU
                            cur_layer = profile_cost[g][str(int(mp.item()))][layer_id]
                        else: # A10 A6000 GPU
                            cur_layer = bs * profile_cost[g][str(int(mp.item()))][layer_id]
                            
                    elif layer_type == "transformer_layer":
                        cur_layer = bs * profile_cost[g][str(int(mp.item()))][layer_id]
                    else:
                        cur_layer = 0
                    cost_e_arr[i][layer_id] = cur_layer
                    
        cost_e_arr = torch.from_numpy(np.stack(cost_e_arr, axis=0))
        cost_e_arr = torch.mean(cost_e_arr, dim=0) 
        cost_e[g] = cost_e_arr
    
    return cost_e


def cost_all_reduce_embedding(args:Namespace, model_config, cluster_info, parallel_config):
    precision = args.precision
    tp_degree = int(parallel_config["mp"].item())
    dp_degree = int(parallel_config["dp"].item())
    pp_degree = int(parallel_config["pp"].item())
    rank_node_map = parallel_config["rank_node_map"]
    hidden_size = int(model_config["hidden_size"].item())
    vocab_size = int(model_config["vocab_size"].item())
    comm_type = args.comm_type
    gpu_per_node = args.gpu_per_node

    if pp_degree>1:
        # Get communication bandwidth between pipeline stage 0 and -1
        for i in range(dp_degree):    
            # get the slowest mp gpu connection
            slowest_bandwidth = np.inf
            for k in range(tp_degree):    
                rank_cur = axis2rank(axis=(0,i,k), mp_deg=tp_degree, dp_deg=dp_degree, pp_deg=pp_degree)
                rank_peer = axis2rank(axis=(pp_degree-1,i,k), mp_deg=tp_degree, dp_deg=dp_degree, pp_deg=pp_degree)
                node_cur = rank_node_map[rank_cur]
                node_peer = rank_node_map[rank_peer]
                
                if node_cur != node_peer: # use inter-node bandwidth
                    cur_bandwidth = min(cluster_info[node_cur][0], cluster_info[node_peer][0])
                else: # use intra-node bandwidth
                    cur_bandwidth = cluster_info[node_cur][1]
                if cur_bandwidth < slowest_bandwidth:
                    slowest_bandwidth = cur_bandwidth
        
        # if dp_degree<gpu_per_node, we assume the bandwidth is shared by all dp_degree
        # else, we assume the bandwidth is shared by all gpu_per_node
        band_width = slowest_bandwidth/min(dp_degree, gpu_per_node) 
        embedding_syn_cost = 2*(2-1)*(hidden_size*vocab_size*precision)/(2*band_width)/tp_degree
        return embedding_syn_cost.item()
    else:
        return 0
        

def dp_cost(args:Namespace, config, cluster_info, model_config, parallel_config, partition, num_mb, _layer=None, gpu_type_lst=None, dp_method="dp"):
    h = model_config["hidden_size"]
    n = model_config["num_layers"]
    v = model_config["vocab_size"]
    rank_node_map = parallel_config["rank_node_map"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]
    gpu_per_node = args.gpu_per_node
    comm_type = args.comm_type
    dp_method = dp_method
    precision = args.precision
    num_mb = num_mb
    
    _num_layer = len(_layer)
        
    # First translate to deepspeed partition form
    ds_partition = [0]
    for i in range(len(partition)):
        ds_partition.append(ds_partition[-1]+partition[i])
    assert ds_partition[-1] == _num_layer
    assert len(ds_partition) == pp + 1

    counted = False
    # debug
    # print(f"tp: {int(mp.item())}, dp: {int(dp.item())}, pp: {int(pp.item())}, comm_type: {comm_type}")
    
    param_count = 0    
    for layer_id in range(ds_partition[0], ds_partition[1]):
        layer_type = _layer[layer_id]
        if layer_type == "embedding_layer" or layer_type == "post_process":
            if not counted:
                counted = True
                param_count += (h*v)
        elif layer_type == "transformer_layer" or layer_type == "encoder" or layer_type == "decoder":
            param_count += ((12 * h ** 2)+20800) / mp
    
    # print(f" param_count: {param_count}")
    
    # Get communication bandwidth of pipeline stage 0
    dp_cost_list = []
    nodelist = []
    
    for i in range(int(pp.item())):
        for j in range(int(mp.item())):
            bandwidth_lst = []
            gpu_m_lst = []
            for k in range(int(dp.item())):
                rank_cur = axis2rank(axis=(0,k,j), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                node_cur = rank_node_map[int(rank_cur.item())]
                rank_next = axis2rank(axis=(0,(k+1)%(dp.item()),j), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                node_next = rank_node_map[int(rank_next.item())]
                nodelist.append(int(node_cur))
                
                # intra-node
                if node_cur == node_next:
                    connectivity = cluster_info[node_cur][1]

                else:
                    cur_idx = 0
                    next_idx = 0
                    
                    cur_node = cluster_info[node_cur][cur_idx]
                    next_node = cluster_info[node_next][next_idx]    
                    connectivity = min(cur_node, next_node)
                        
                bandwidth_lst.append(connectivity)
            
        # get slowest of bandwidth
        bandwidth = min(bandwidth_lst)
        
        # precision = 16
        cost = 0.0
        if dp_method in ["dp", "zero0"]:
            # All-Reduce cost: 2(n-1)M / nB
            cost = 2 * (int(dp.item()) - 1) * (param_count * precision) / (int(dp.item()) * bandwidth)
        elif dp_method in ["zero1","fsdp"]:
            # Reduce-Scatter, All-Gather cost: (n-1)M / nB
            
            rs = ( (int(dp.item()) - 1) * (param_count * precision) / (int(dp.item()) * bandwidth) )
            ag = ( (int(dp.item()) - 1) * (param_count * precision) / (int(dp.item()) * bandwidth) )
            cost = rs + ag
        elif dp_method == "zero2":
            zero2_precision = 16
            reduce = ( (int(dp.item()) - 1) * (param_count * zero2_precision) / (int(dp.item()) * bandwidth) ) * num_mb
            # print(f"reduce: {reduce}")
            ag = ( (int(dp.item()) - 1) * (param_count * zero2_precision) / (int(dp.item()) * bandwidth) )
            # print(f"allgather:{ag}")
            cost = reduce + ag
        elif dp_method == "zero3":
            zero3_precision = 32
            rs = ( (int(dp.item()) - 1) * (param_count * zero3_precision) / (int(dp.item()) * bandwidth) ) * num_mb
            cost = rs
        
        dp_cost_list.append(cost)
        
    return ds_partition, dp_cost_list


def predict(args:Namespace, config, gbs, mbs, cluster_info, model_config, zerosearch_config, oth, node_placement, dp_method, exhaustive_dict, node_map):
    L = int(model_config["num_layers"])
    model_type = model_config["type"]
    cost = torch.zeros(1,)
    M, N = config.shape
    config = np.asarray(config)
    dp_method = dp_method
    exhaustive = args.exhaustive
    exhaustive_dict = exhaustive_dict
       
    if np.all(config == -1):
        rank_map = defaultdict(list)
        rank_node_map = dict()

        tp_degree = oth["tp_deg"]
        dp_degree = oth["dp_deg"]
        pp_degree = oth["pp_deg"]                   
        
        # infer a GPU rank map                
        counter = 0 
        for j in range(N):
            for k in range(M):
                # TODO: bad code here, config counts from 1
                rank_map[j].append(counter)
                rank_node_map[counter] = j
                counter += 1
    
    # valid config, inferred from sa
    else:
        config = torch.from_numpy(config)
        pp = torch.max(config).float()
        
        # infer rank_map: given node name, returns the global mapped rank(int) in (pp, dp, mp) order
        # rank_node_map: given rank, returns the node
        rank_map = defaultdict(list)
        rank_node_map = dict()
           
        tp_degree = oth["tp_deg"]
        dp_degree = oth["dp_deg"]
        pp_degree = oth["pp_deg"]                  
        
        rank_counter = np.zeros(int(pp.item()))
            
        # infer a GPU rank map
        for j in range(N):
            for k in range(M):
                # TODO: bad code here, config counts from 1
                cur_pp = int(config[k][j] - 1)
                rank_map[j].append(int((rank_counter[cur_pp] + cur_pp * tp_degree * dp_degree).item()))
                rank_node_map[int((rank_counter[cur_pp] + cur_pp * tp_degree * dp_degree).item())] = j
                rank_counter[cur_pp] += 1
            
    num_mb = gbs / (dp_degree * mbs)
    # print(f"gbs:{gbs}, dp_degree:{dp_degree}, mbs: {mbs}, num_mb: {num_mb}")
    parallel_config = {"num_mb":num_mb, "mp" : tp_degree, "dp" : dp_degree, "pp" : pp_degree, "micro_bs" : mbs, "rank_map" : rank_map, "rank_node_map": rank_node_map, "dp_method":dp_method}
    # print(f"parallel_config: {parallel_config}")
    pp_degree = int(pp_degree.item())
    _layer = get_layer_type(model_type=model_type, n=L, pp=pp_degree)
    cost_e = get_cost_e(model_config=model_config, parallel_config=parallel_config, profile_cost=zerosearch_config, _layer=_layer, model_type=model_type)
    cost_c, layer_type = get_cost_c(args, cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, _layer=_layer)
    gpu_type_lst = get_gpu_for_stage(pp_degree, N, node_placement, node_map) # assign GPUs for pp degrees
    # print(f"cost_c: {cost_c} \nnode_placement: {node_placement} \ngpu_type_lst: {gpu_type_lst}")
    # print(f"gpu_type_lst: {gpu_type_lst}")
    # print(f"cost_e: {cost_e}")
    partition=[]
    if model_type in ['gpt2XL','llama2_13B',"llama2_13B_mini"]:
        if exhaustive:
            # print(f"here L : {L}")
            if args.search_method == "minmax":
                partition, stage_comp_time_lst, _, _, stage_for_send_time_lst, stage_back_send_time_lst  = explain_minmax(L+2, cost_e, np.asarray(cost_c), pp_degree, gpu_type_lst, exhaustive_dict["partition"])
                
                pipecost_last, stage_wise_cost_lst = schedule(pp_degree, 
                                                        num_mb, stage_comp_time_lst, 
                                                        stage_for_send_time_lst, 
                                                        stage_back_send_time_lst,
                                                        dp_method)
                
                # assert False, "Minmax Done!"
                # print(f"pipecost_last: {pipecost_last}")
            elif args.search_method == "ga":
                partition, pipecost_last, stage_wise_cost_lst = ga(L+2, parallel_config, cost_e, np.asarray(cost_c), gpu_type_lst)
        else:
            if args.search_method == "minmax":
                partition, stage_comp_time_lst, _, _, stage_for_send_time_lst, stage_back_send_time_lst  = minmax(L+2, cost_e, np.asarray(cost_c), pp_degree, gpu_type_lst)
                pipecost_last, stage_wise_cost_lst = schedule(pp_degree, 
                                                        num_mb, stage_comp_time_lst, 
                                                        stage_for_send_time_lst, 
                                                        stage_back_send_time_lst,
                                                        dp_method)
            elif args.search_method == "ga":
                partition, pipecost_last, stage_wise_cost_lst = ga(L+2, parallel_config, cost_e, np.asarray(cost_c), gpu_type_lst)    
        
        is_oom, gpumem = EstimatePeakMemory(args, partition, model_config, parallel_config, layer_type, gpu_type_lst, gbs, num_mb, dp_method)
        # print(f"gpumem: {gpumem}")
        # delete is zero oom . 어차피 zero가 dp_method로 표현될 수 있기 때문
        
    # translate to ds form, add data parallelism cost
    _, dp_cost_list = dp_cost(args, config, cluster_info, model_config, parallel_config, 
                        partition, num_mb, _layer, gpu_type_lst, dp_method)
    
    if model_type != "T5":
        all_reduce_embedding_cost = cost_all_reduce_embedding(args, model_config, cluster_info, parallel_config)
    else:
        all_reduce_embedding_cost = 0

    end2end_stage_latency=[]
    for i in range(len(stage_wise_cost_lst)):
            end2end_stage_latency.append(stage_wise_cost_lst[i] + dp_cost_list[i])
    cost_last = max(end2end_stage_latency) + all_reduce_embedding_cost
    
    max_latency = max(end2end_stage_latency)
    max_latency_index = end2end_stage_latency.index(max_latency)
    
    dp_side_cost_last = dp_cost_list[max_latency_index]

    if exhaustive is True:
        print(f"dp method: {dp_method}")
        print(f"partition: {partition}")
        print(f"estimated time / pp latency / dp cost / allreduce embedding")
        print(f"{cost_last.item():.4f}, {pipecost_last.item():.4f}, {dp_side_cost_last.item():.4f}, {all_reduce_embedding_cost:.4f}")
        assert False, "Done!"

    return rank_map, partition, cost_last, pipecost_last, dp_side_cost_last, all_reduce_embedding_cost, is_oom, gpumem
    

def EstimatePeakMemory(args:Namespace, partition, model_config, parallel_config, layer_type, gpu_type_lst, gbs, num_mb, dp_method):
    h = model_config["hidden_size"] 
    v = model_config["vocab_size"]
    s = model_config["sequence_length"]
    a = model_config["num_attention_heads"]
    d = model_config["ffn_hidden_size"]
    tp = parallel_config["mp"]
    pp = parallel_config["pp"] 
    dp = parallel_config["dp"]
    b = parallel_config["micro_bs"]
    L = model_config["num_layers"]
    model_type = args.type
    gbs = gbs
    num_mb = num_mb
    dp_method = dp_method
    M = args.gpu_per_node
    additional_buffer_factor=1.5
    error_percent = 1.0
    
    # debugging
    # print(f"{model_type} layer info\nh:{h}, v:{v}, s:{s}, a:{a}, d:{d}, b:{b}, dp_method: {dp_method}")
    memory = []
    memory_side = {"weight":[], "activation":[]}
    oom_list = []
    p = pp
    if num_mb > pp:
        p = pp
    else:
        p = num_mb
    st = 0
    en = 0
    
    # debugging
    # print(f"partition: {partition}")
    gpu_memory = {"A10":22.20, "RTX3090":23.06, "A6000":48, "A100":48}
    for j, stage in enumerate(partition):
        oom = False
        st = 0 + en 
        en += stage
        param_count = 0 # unit: bytes
        activation = 0 # unit: bytes
        major = 0
        # pipeline_buffer = 0
        for i in range(st, en):
            if layer_type[i] == "embedding_layer" :
                param_count += ( h * v ) / tp
                if model_type == "llama2_13B":
                    pass
                else:
                    activation += ( s * b * h * pp * p) / tp
                
                # debugging
                # activation += ( s ) / tp
                # print(f"[{j}]: layer-{st} s: {s}, b: {b}, h: {h}, pp: {pp}, p: {p}, tp: {tp}")
                # print(f"[{j}]: layer-{st} layer_type: {layer_type[i]} param_count:{int(param_count.item())} activation: {int(activation.item())}")
            elif layer_type[i] == "transformer_layer" or layer_type[i] == "encoder" or layer_type[i] == "decoder":
                if model_type == "llama2_13B":
                    kv_value=8
                    m=2
                    param_count += (h*1) + (h*s)*3 / tp + (h*1) + (h*d*3) / tp
                    
                    # debugging
                    # activation += (s * b * p * h ) * (10 + ( 24 / tp ) + 5 * (a * s) / (h * tp) )  # tensor + sequence
                    activation += (10*s*b*h + (16/tp)*s*b*h + (2*s*b*d/tp) + (5/tp)*kv_value*s*s*b) * (1 + (pp-1)/(pp*m))
                    # print(f"activation: stage {j}th layer {i}th {activation.item()/1024/1024/1024}")
                else:
                    param_count += ( 12 * h ** 2 ) / tp
                    activation += (s * b * p * h ) * (10 + ( 24 / tp ) + 5 * (a * s) / (h * tp) )  # tensor + sequence 
                
                # debugging
                # activation += (s * b * p * h ) * (10 + ( 24 / tp ) + 5 * (a * s) / (h * tp) )  # tensor + sequence
                # activation += (s * b * h  ) * (10 + ( 24 / tp ) + 5 * (a * s) / (h * tp) )  # tensor + sequence
                # if i == 0:
                    # pipeline_buffer += mbs * seq_len * h # BLH
                # print(f"[{j}]: layer_type: {layer_type[i]} param_count:{int(param_count.item())} activation: {int(activation.item())}")    
            elif layer_type[i] == "post_process":
                param_count += ( h * v ) / tp
                if model_type == "llama2_13B":
                    pass
                else:
                    activation += (4 * s * b * h ) / tp + ( 4 * s * b * v ) / tp
                
                # debugging
                # print(f"[{j}]: layer_type: {layer_type[i]} param_count:{int(param_count.item())} activation: {int(activation.item())}")
        
        # debugging
        # print(f"stage {j}th actication: {activation} ")    
        memory_side["activation"].append(activation)
        if dp.item() > 1:
            if dp_method in ['fsdp']:
                weight = param_count * (4 + (16 / dp))
                
            elif dp_method in ["zero1"]:
                weight = 4 * param_count + ( 16 * param_count / dp)
                
            elif dp_method in ["zero2"]:
                weight = 4 * param_count + ( 16 * param_count / dp)
            elif dp_method in ["zero3"]:
                transformer_layer = ( h * v  )
                weight = 4 * transformer_layer * M * additional_buffer_factor + (18 * param_count / dp)
            else:
                weight = param_count * 18                 
        else:           
            weight = param_count * 18
            
        # debugging     
        # print(f"weight memory: {weight}")
        memory_side["weight"].append(weight)
        total_mem = weight + activation
        estimated_memory = total_mem / 1024 /1024 / 1024
        
        # debugging
        # print(f"num sublayer: {stage} major: {estimated_memory.item():.2f}")
        # print(f"estimated memory: {estimated_memory.item():.4f}")
        
        gpumem = estimated_memory * error_percent
        memory.append(estimated_memory)
        oom_list.append(oom)
        if args.exhaustive:
            pass
        else:
            if estimated_memory * error_percent > gpu_memory[gpu_type_lst[j]]:
                oom = True
                # TODO: return not search
                return oom, gpumem
            else:
                pass
        
    gpumem = max(memory)
    if args.exhaustive:
        i = memory.index(gpumem)
        print(f"activation: {memory_side['activation']}")
        print(f"{i}th gpu memory usage: {gpumem.item():.2f}")
        print(f"{i}th gpu weight memory usage: {memory_side['weight'][i].item()/(1024*1024*1024):.2f}")
        print(f"{i}th gpu activation memory usage: {memory_side['activation'][i].item()/(1024*1024*1024):.2f}")
    
    # assert False, "Done!"
    
    return oom, gpumem
    


def get_layer_type(model_type, n, pp):
    _layer = ["embedding_layer"]
    if model_type != "T5":
        for i in range(n):
            _layer.append("transformer_layer")
        _layer.append("post_process")

    else:
        for i in range(int(n/2)):
            _layer.append("encoder")
        _layer.append("embedding_layer")
        for i in range(int(n/2)):
            _layer.append("decoder")

    return _layer
