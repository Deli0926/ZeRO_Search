from argparse import Namespace
from itertools import permutations

def get_gpu_for_stage(pp, N, node_placement, node_map):
    # if i have 3 heterogenous gpu, how to do? TODO
    gpu_for_stage = []    
    if pp == 1: # bound lower gpu
        if len(node_map.keys()) == 1: # homogeneous
            gpu_for_stage.append(node_placement[0])
        else: # heterogeneous
            if 'A10' in node_map.keys(): 
                gpu_for_stage.append('A10') # TODO: must be lowest gpu type
            else:
                gpu_for_stage.append('RTX3090') # TODO: must be lowest gpu type               
    else:
        for stage in range(pp):
            if pp < N:
                stage_per_node = N/pp
                gpu = 'A6000'
                for node_idx in range(int(stage_per_node*stage), int(stage_per_node*(stage+1))):
                    if node_placement[node_idx] == 'A10':
                        gpu = 'A10'
                    elif node_placement[node_idx] == 'RTX3090':
                        gpu = 'RTX3090'
                gpu_for_stage.append(gpu)
            elif pp > N:
                node_per_pp = pp/N
                node_idx = int(stage//node_per_pp)
                gpu_for_stage.append(node_placement[node_idx])
            else: # pp == N
                node_idx = stage
                gpu_for_stage.append(node_placement[node_idx])
    return gpu_for_stage
                        


def get_cluster_list(args:Namespace, node_map):
    # Returns cluster list for the given model type
    # hetero_node 2 = 0: A10, 1: A6000
    # hetero_node 3 = 0: A10, 1: RTX3090, 2: A6000
    cluster_info={}
    node_map = node_map
    num_node = args.num_node
    pareto=args.pareto
    model_type=args.type
    
    if pareto: # free option
        num_c = 0
        cluster_combinations = []
        for A in range(0, num_node+1):
            for B in range(0, num_node+1):
                cluster = {}
                # if A + B <= num_node:
                for i in range(A + B):
                    cluster[i] = '0'
                for i in range(A):
                    cluster.update({i:'1'})
                if len(cluster.keys())>0:
                    cluster_combinations.append(cluster)
                num_c += 1
                print(f"[{num_c}] {cluster}")
        # print(f"Number of clusters combinations: {num_c}")
        # assert False
        return cluster_combinations
    if model_type in ["gpt2XL","llama2_13B","llama2_13B_mini"]:
        if len(node_map.keys()) > 1:
            # return ['1','1','0','0',...]
            cluster_info = [ str(list(node_map.keys()).index(i)) for i in node_map for _ in range(node_map[i]) ]
        else: # homogeneous
            cluster_info = [ '0' for i in range(0, num_node)]
        cluster_combinations = [cluster_info]
        return cluster_combinations


def device_placement(node_map, cluster):
    D = cyclic_permutation(cluster)
    if len(node_map.keys()) == 1:
        return [D[0]]
    else:
        return D

def cyclic_permutation(l):
    """
    Returns all cyclic permutations of the given list
    """
    permutations = []
    count = 0
    cluster_type_set = set(l)
    
    is_homo = len(cluster_type_set)
    
    if is_homo == 1:
        return [l]
    for i in range(len(l)):
        permutations.append(l[i:] + l[:i])
        count += 1
    # print(f"Number of node placement: {count}")
    return permutations


def msp(items):
  '''Yield the permutations of `items` where items is either a list
  of integers representing the actual items or a list of hashable items.
  The output are the unique permutations of the items given as a list
  of integers 0, ..., n-1 that represent the n unique elements in
  `items`.

  Examples
  ========

  >>> for i in msp('xoxox'):
  ...   print(i)

  [1, 1, 1, 0, 0]
  [0, 1, 1, 1, 0]
  [1, 0, 1, 1, 0]
  [1, 1, 0, 1, 0]
  [0, 1, 1, 0, 1]
  [1, 0, 1, 0, 1]
  [0, 1, 0, 1, 1]
  [0, 0, 1, 1, 1]
  [1, 0, 0, 1, 1]
  [1, 1, 0, 0, 1]

  Reference: "An O(1) Time Algorithm for Generating Multiset Permutations", Tadao Takaoka
  https://pdfs.semanticscholar.org/83b2/6f222e8648a7a0599309a40af21837a0264b.pdf
  '''

  def visit(head):
      (rv, j) = ([], head)
      for i in range(N):
          (dat, j) = E[j]
          rv.append(dat)
      return rv

  u = list(set(items))
  E = list(([u.index(i) for i in items]))
  N = len(E)
  # put E into linked-list format
  (val, nxt) = (0, 1)
  for i in range(N):
      E[i] = [E[i], i + 1]
  E[-1][nxt] = None
  head = 0
  afteri = N - 1
  i = afteri - 1
  yield visit(head)
  while E[afteri][nxt] is not None or E[afteri][val] < E[head][val]:
      j = E[afteri][nxt]  # added to algorithm for clarity
      if j is not None and E[i][val] >= E[j][val]:
          beforek = afteri
      else:
          beforek = i
      k = E[beforek][nxt]
      E[beforek][nxt] = E[k][nxt]
      E[k][nxt] = head
      if E[k][val] < E[head][val]:
          i = k
      afteri = E[i][nxt]
      head = k
      yield visit(head)