"""
Portions of this code adapted from the 'AMP' project (https://github.com/DachengLi1/AMP). 
@article{li2022amp,
  title={AMP: Automatically Finding Model Parallel Strategies with Heterogeneity Awareness},
  author={Li, Dacheng and Wang, Hongyi and Xing, Eric and Zhang, Hao},
  journal={arXiv preprint arXiv:2210.07297},
  year={2022}
}
"""

from collections import defaultdict
from argparse import Namespace
import pandas as pd

# returns the rank to axis. If pp_deg=dp_deg=mp_deg=2, rank 3 gives (0,1,1).
# This is deepspeed method
def rank2axis(rank, mp_deg, dp_deg, pp_deg):
    pp = rank // (mp_deg * dp_deg)
    remainder = rank % (mp_deg * dp_deg)

    dp = remainder // (mp_deg)
    remainder = remainder % mp_deg

    mp = remainder

    return (pp, dp, mp)

# returns the axis to rank. If pp_deg=dp_deg=mp_deg=2, (0,1,1) gives 3
def axis2rank(axis, mp_deg, dp_deg, pp_deg):
    pp, dp, mp = axis
    return mp + mp_deg * dp + (mp_deg * dp_deg) * pp

def factor(N, upper=None):
    if upper is None:
        upper = N
    ret = []
    for i in range(1, upper+1):
        if N % i == 0:
            ret.append(i)
    return ret

def get_dp_method(args:Namespace):
    # ZeRO, overlap_comm --framework, --overlap_comm
    if args.framework == "m":
        # search only Megatron-LM framework
        dp_method = ["dp", "fsdp"]
    elif args.framework == "d":
        # search only DeepSpeed framework
        dp_method = ["zero0", "zero1", "zero2", "zero3"]
    elif args.framework == "both":
        # search both framework
        dp_method = ["dp", "fsdp", "zero0", "zero1", "zero2", "zero3"]
    elif args.framework == "default":
        # search both framework with no redundancy
        dp_method = ["dp", "fsdp", "zero2", "zero3"]
    else:
        assert False, "you have to chooese framework among m, d, both, and default"

    if args.exhaustive:
        print("DEBUG: IS changed?")
        dp_method = "zero3"
        overlap = False
    return dp_method

def no_placement_strategy(M, N, gbs, known, num_layers):
    # known is 3d parallelism combination (tp, pp, dp)
    if known is None:
        known = defaultdict(list)
        ele_count = 0
        W = M * N # world size
        for h in factor(min(M, 4)): # mp, only max 4 is supported
            # h: tp degree 
            # w: dp degree
            assert M*N % h == 0
            remain = M*N // h
            for w in factor(remain): # dp
                pp_degree = M*N / (h*w)
                # if pp_degree is not int
                if pp_degree != int(pp_degree):
                    continue
                if (W / pp_degree) % w != 0:
                    continue
                if gbs % (w) != 0:
                    continue
                if pp_degree > num_layers:
                    continue
                for mbs in factor(gbs // w):
                    ele_count += 1
                    # known.keys() == mbs
                    known[mbs].append((h, w))
                    
                    
    if len(known.keys()) == 0:
        return None

    mbs = list(known.keys())[0]
    (h, w) = known[mbs].pop(0)
    if len(known[mbs]) == 0:
       known.pop(mbs, None)

    return h, w, mbs, known

def no_placement_strategy_with_zero(args:Namespace, M, N, gbs, known, num_layers, dp_method, exhaustive_dict):
    # known is 3d parallelism combination (tp, pp, dp)
    # known: 처음에만 만드는 것
    W = M * N # world size
    search_total = True
    if known is None:
        known = defaultdict(list)
        ele_count = 0
        if args.exhaustive:
            mbs = exhaustive_dict["mbs"]
            tp = exhaustive_dict["tp"]
            dp = exhaustive_dict["dp"]
            pp = exhaustive_dict["pp"]
            i="zero"+str(exhaustive_dict["zero"])
            #i = "zero1"
            ele_count += 1
            print(f"mbs:{mbs}, tp:{tp}, dp:{dp}, pp:{pp}, i:{i}")
            return i, tp, dp, pp, mbs, known
        else:
            for i in dp_method:
                # dp_method: [dp, fsdp, zero2, zero3] _3d는 한번 만들어두고 복사해서 쓰면됨
                if search_total:
                    r = W
                else:
                    r = min(M, args.gpu_per_node)
                for h in factor(args.gpu_per_node): # mp(=tp), only max 4 is supported
                    # h: tp degree 
                    # w: dp degree
                    assert M*N % h == 0
                    remain = M*N // h
                    
                    if i == "zero3":
                        if h > 1:
                            continue 
                        
                    for w in factor(remain): # dp
                        pp_degree = M*N / (h*w)
                        if i == "zero2" and pp_degree > 1:
                            continue
                        if i == "zero3" and pp_degree > 1:
                            continue
                        # if pp_degree is not int
                        if pp_degree != int(pp_degree):
                            continue
                        if (W / pp_degree) % w != 0:
                            continue
                        if gbs % (w) != 0:
                            continue
                        if pp_degree > num_layers:
                            continue
                        for mbs in factor(gbs // w):
                            ele_count += 1
                            known[mbs].append((i, h, w))
                # print(f"known: {known}")
    if len(known.keys()) == 0:
        return None

    if args.search_space:
        search_space = {}
        count=1
        for mbs in known:
            for j in known[mbs]:
                search_dict = {}
                (i, tp, dp) = j
                pp = int(W/ (tp * dp))
                search_dict["mbs"] = mbs
                search_dict["tp"] = tp
                search_dict["pp"] = pp
                search_dict["dp"] = dp
                search_dict["dp_method"] = i
                search_space[count] = search_dict
                count = count + 1
                
        search_df = pd.DataFrame(search_space)
        search_df_T = search_df.T
        # search_df = search_df.sort_values(by="dp_method", ignore_index=True)
        mbs_group_count = search_df_T.groupby(["tp","pp","dp","dp_method"])["mbs"].count()
        tp_group_count = search_df_T.groupby(["tp","pp","dp","dp_method"])["tp"].count()
        # print(grouped_count)
        mbs_group_count.to_csv(f"main_logs/_N{N}_gbs{gbs}_mbs_group_count.csv")
        tp_group_count.to_csv(f"main_logs/_N{N}_gbs{gbs}_tp_group_count.csv")
        _3d = search_df_T[(search_df_T['dp_method'] == 'zero0')]
        
        print(f"_3d count: {_3d}, \n all combination: {search_df_T.count()}")
        
        search_df_T.to_csv(f"main_logs/_N{N}_gbs{gbs}_Total{search_total}_search.csv")
            
        assert False, "Done!"
        # df.to_csv("_search.csv")

    mbs = list(known.keys())[0]
    (i, h, w) = known[mbs].pop(0)
    if len(known[mbs]) == 0:
       known.pop(mbs, None)
    # print(f"{i}, tp: {h}, dp: {w}")
    return i, h, w, mbs, known

def no_placement_strategy_with_zero_and_overlap(M, N, gbs, known, num_layers, dp_method):
    # known is 3d parallelism combination (tp, pp, dp)
    # known: 처음에만 만드는 것
    if known is None:
        known = defaultdict(list)
        ele_count = 0
        W = M * N # world size
        for i in dp_method:
            # dp_method: [dp, fsdp, zero2, zero3] _3d는 한번 만들어두고 복사해서 쓰면됨
            
            for h in factor(min(M, 4)): # mp(=tp), only max 4 is supported
                # h: tp degree 
                # w: dp degree
                assert M*N % h == 0
                remain = M*N // h
                
                if i == "zero3":
                    if h > 1:
                        continue 
                    
                for w in factor(remain): # dp
                    pp_degree = M*N / (h*w)
                    if i == "zero2" and pp_degree > 1:
                        continue
                    
                    if i == "zero3" and pp_degree > 1:
                        continue
                    # if pp_degree is not int
                    if pp_degree != int(pp_degree):
                        continue
                    if (W / pp_degree) % w != 0:
                        continue
                    if gbs % (w) != 0:
                        continue
                    if pp_degree > num_layers:
                        continue
                    for mbs in factor(gbs // w):
                        ele_count += 1
                        # known.keys() == mbs
                        known[mbs].append((i, h, w))
            # print(f"known: {known}")
                    
                    
    if len(known.keys()) == 0:
        return None

    mbs = list(known.keys())[0]
    (i, h, w) = known[mbs].pop(0)
    if len(known[mbs]) == 0:
       known.pop(mbs, None)

    return i, h, w, mbs, known

def get_node_map(args:Namespace):
    
    # valid test
    if args.node_type is None:
        args.node_type = ["A10"]
    #A10 8개만 있다고 가정
    '''
    if args.node_type_num_node is None:
        args.node_type_num_node = ['8']
    '''
    if args.node_type_num_node is None:
        if args.num_node is None: args.node_type_num_node = ['8']
        else: args.node_type_num_node = [str(args.num_node)]
    node_type = args.node_type
    node_type_num_node = args.node_type_num_node
    
    # valid test
    if len(node_type) != len(node_type_num_node):
        # print(f"node_type: {len(node_type)}")
        # print(f"node_type_num_node: {len(node_type_num_node)}")
        if args.pareto:
            pass
        else:
            assert False, "node_type size must be same as node_type_num_node size"
    
    node_type_num_node = [int(n) for n in node_type_num_node]
    node_map = dict(zip(node_type, node_type_num_node))
    n = sum(node_type_num_node)
    if args.num_node != n:
        args.num_node = n
    
    # assert False, f"node_map: {node_map}, {args.num_node}"
    return node_map