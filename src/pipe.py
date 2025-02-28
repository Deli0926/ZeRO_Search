import torch
import time
import numpy as np
import copy

from stage import PPGroup
from pygad import GA
import logging
from datetime import datetime

class Stage:
    def __init__(self):
        self.comm_time = 0.
        self.comp_time = 0.
        self.for_send_time = 0.
        self.back_send_time = 0.

    def set_comp_time(self, comp_time):
        self.comp_time = comp_time

    def set_comm_time(self, comm_time):
        self.comm_time = comm_time
    
    def set_for_send_time(self, for_send_time):
        self.for_send_time = for_send_time
    
    def set_back_send_time(self, back_send_time):
        self.back_send_time = back_send_time

    def get_comp_time(self):
        return self.comp_time
    
    def get_comm_time(self):
        return self.comm_time
    
    def get_for_send_time(self):
        return self.for_send_time

    def get_back_send_time(self):
        return self.back_send_time

    def get_stage_time(self):
        return self.comm_time+self.comp_time


def minmax(num_layer, cost_e, cost_c, pp_degree, gpu_type_lst):

    # print(f"pp_degree: {pp_degree}")
    num_balanced_layer = num_layer // pp_degree
    partition = []
    for i in range(pp_degree):
        partition.append(num_balanced_layer)
    rest = int(num_layer - (num_balanced_layer * pp_degree))
    for i in range(rest):
        partition[i-1] += 1

    partition_history = []
    partition_history.append(partition[:])

    last_max_latency = 1000000
    counted = False
    # print(f"partition_history: {partition_history}")
    while(1):
        stage_latency = get_stage_latency(partition, cost_e, cost_c, gpu_type_lst)
        stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
        # print(f"stage_time_lst: {stage_time_lst}")
        max_latency = max(stage_time_lst)
        if max_latency > last_max_latency:
            if counted:
                partition[max_latency_index] += 1
                partition[min_latency_index] -= 1
                stage_latency = get_stage_latency(partition, cost_e, cost_c, gpu_type_lst)
                break
        if max_latency == last_max_latency:
            if counted and partition in partition_history[:-1]:
                partition[max_latency_index] += 1
                partition[min_latency_index] -= 1
                stage_latency = get_stage_latency(partition, cost_e, cost_c, gpu_type_lst)
                break
        last_max_latency = max_latency
        max_latency_index = stage_time_lst.index(max_latency)

        min_latency = min(stage_time_lst)
        min_latency_index = stage_time_lst.index(min_latency)

        if (max_latency_index == 0 or max_latency_index == pp_degree-1) and partition[max_latency_index] == 2:
            if counted:
                partition[max_latency_index] += 1
                partition[min_latency_index] -= 1
            break
        if partition[max_latency_index]>1:
            partition[max_latency_index] -= 1
            partition[min_latency_index] += 1
            counted=True
            partition_history.append(partition[:])
        else: # no layers to substract
            break
    
    stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
    stage_comp_time_lst = [stage.get_comp_time() for stage in stage_latency]
    # print(f"stage_comp_time_lst: {stage_comp_time_lst}")
    stage_comm_time_lst = [stage.get_comm_time() for stage in stage_latency]
    stage_for_send_time_lst = [stage.get_for_send_time() for stage in stage_latency]
    stage_back_send_time_lst = [stage.get_back_send_time() for stage in stage_latency]

    return partition, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst


def explain_minmax(num_layer, cost_e, cost_c, pp_degree, gpu_type_lst, partition):
    
    num_balanced_layer = num_layer // pp_degree
    # partition = partition
    partition = []    
    print(f"pp_degree: {pp_degree}")
    for i in range(pp_degree):
        partition.append(num_balanced_layer)
    rest = int(num_layer - (num_balanced_layer * pp_degree))
    for i in range(rest):
        partition[i-1] += 1
    
    partition_history = []
    partition_history.append(partition[:])
    
    last_max_latency = 1000000
    counted = False
    while(1):
        # s = time.time()
        stage_latency = get_stage_latency(partition, cost_e, cost_c, gpu_type_lst)
        stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
        max_latency = max(stage_time_lst)
        # e = time.time()
        # print(f"get max stage latency runtime: {e-s}")
        stage_time_lst_a = [ round(stage_time_lst[i].item(),5) for i in range(len(stage_time_lst))]
        # max_latency_index = stage_time_lst.index(max_latency)
        # print(f"stage_time_lst : {stage_time_lst_a}, \npartition: {partition} \nmax_latency: {max_latency*1000:.3f} \nstd: {np.std(stage_time_lst)*1000:.3f}")
        # break
        if max_latency > last_max_latency:
            if counted:
                partition[max_latency_index] += 1
                partition[min_latency_index] -= 1
                stage_latency = get_stage_latency(partition, cost_e, cost_c, gpu_type_lst)
                # print(f"quit 1")
                
                break
        if max_latency == last_max_latency:
            if counted and partition in partition_history[:-1]:
                partition[max_latency_index] += 1
                partition[min_latency_index] -= 1
                stage_latency = get_stage_latency(partition, cost_e, cost_c, gpu_type_lst)
                # print(f"quit 2")
                
                break
        last_max_latency = max_latency
        max_latency_index = stage_time_lst.index(max_latency)
        # print(f"max index: {max_latency_index}")
        min_latency = min(stage_time_lst)
        min_latency_index = stage_time_lst.index(min_latency)

        if (max_latency_index == 0 or max_latency_index == pp_degree-1) and partition[max_latency_index] == 2:
            if counted:
                partition[max_latency_index] += 1
                partition[min_latency_index] -= 1
            # print(f"quit 3")
            break
        if partition[max_latency_index]>1:
            partition[max_latency_index] -= 1
            partition[min_latency_index] += 1
            counted=True
            partition_history.append(partition[:])
        else: # no layers to substract
            # print(f"quit 4")
            break
        # print(f"2 max_latency: {max_latency}")
    # print(f"max index: {max_latency_index}")
    stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
    stage_comp_time_lst = [stage.get_comp_time() for stage in stage_latency]
    stage_comm_time_lst = [stage.get_comm_time() for stage in stage_latency]
    stage_for_send_time_lst = [stage.get_for_send_time() for stage in stage_latency]
    stage_back_send_time_lst = [stage.get_back_send_time() for stage in stage_latency]
    
    
    return partition, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst


def get_stage_latency(partition, cost_e, cost_c, gpu_type_lst):
    
    num_bw_share = 1 # which should be caculated in get_cost_c considering PCIe
    num_stage = len(partition) # PP =1 이면, num_stage = 1

    stage_latency = [Stage() for _ in range(num_stage)]

    if num_stage == 1: # pp = 1
        try:
            stage_latency[0].set_comp_time(sum(cost_e[gpu_type_lst[0]]))
            return stage_latency
        except:
            assert False, "gpu type is not recognized"
        
    # print(f"partition: {partition}") 
    # print(f"num_stage: {num_stage}")       
    for stage in range(num_stage):
        # print(f"stage: {stage}")
        # print(f"cost_e: {cost_e}")
        num_layer_til_last_stage = sum(partition[:stage]) # partition의 i 번째 레이어 개수
        num_layer_til_cur_stage = sum(partition[:stage+1]) # partition의 i+1번째 레이어 개수
        node_idx = stage
        
        cost_e_arr = cost_e[gpu_type_lst[node_idx]]
               
        if stage == 0:
            stage_latency[stage].set_comp_time(sum(cost_e_arr[:num_layer_til_cur_stage]))
            stage_latency[stage].set_for_send_time((cost_c[sum(partition[:stage])][stage]*num_bw_share).item())
        elif stage == num_stage-1:
            stage_latency[stage].set_comp_time(sum(cost_e_arr[num_layer_til_last_stage:num_layer_til_cur_stage]))
            stage_latency[stage].set_back_send_time((cost_c[sum(partition[:stage])][stage-1]*num_bw_share).item())
        else:
            stage_latency[stage].set_comp_time(sum(cost_e_arr[num_layer_til_last_stage:num_layer_til_cur_stage]))
            stage_latency[stage].set_comm_time((cost_c[sum(partition[:stage])][stage-1]*num_bw_share).item())
    
   
    return stage_latency



def schedule(pp_degree, num_mb, stage_comp_time_lst, stage_for_send_time_lst, stage_back_send_time_lst, dp_method):

    ppgroup_cfg = {"num_mb": None,
                   "pp_degree": None,
                   "stage_comp_time_lst": stage_comp_time_lst,
                   "stage_for_send_time_lst": stage_for_send_time_lst,
                   "stage_back_send_time_lst": stage_back_send_time_lst,
                   "dp_method": dp_method
                   }

    if isinstance(num_mb, torch.Tensor):
        ppgroup_cfg["num_mb"] = int(num_mb.item())
    else:
        ppgroup_cfg["num_mb"] = num_mb
    
    if isinstance(pp_degree, torch.Tensor):
        ppgroup_cfg["pp_degree"] = int(pp_degree.item())
    else:
        ppgroup_cfg["pp_degree"] = pp_degree

    if ppgroup_cfg["pp_degree"] == 1:
        cost = num_mb * sum(stage_comp_time_lst)

    else:    
        my_pp_group = PPGroup(**ppgroup_cfg)
        
        my_pp_group.simulate_full_pipeline()
        cost = my_pp_group.get_pipe_cost()

    if not isinstance(cost, torch.Tensor):
        cost = torch.tensor(cost)

    if ppgroup_cfg["pp_degree"] == 1:
        stage_wise_cost_lst = [cost]
    else:
        stage_wise_cost_lst = my_pp_group.get_stagewise_end_time_lst()

    return cost, stage_wise_cost_lst


def exhaustive_partition(num_layer, cost_e1, cost_e2, cost_c, pp_degree, gpu_type_lst):

    s_time = time.time()
    P = compositions(num_layer, pp_degree)
    max_latency = np.inf
    for p in P:
        cur_latency = get_stage_latency(p, cost_e1, cost_e2, cost_c, gpu_type_lst)
        stage_time_lst = [stage.get_comp_time() for stage in cur_latency]
        
        if max(stage_time_lst) < max_latency:
            partition = p[:]
            stage_latency = cur_latency
            max_latency = max(stage_time_lst)

    stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
    stage_comp_time_lst = [stage.get_comp_time() for stage in stage_latency]
    stage_comm_time_lst = [stage.get_comm_time() for stage in stage_latency]
    stage_for_send_time_lst = [stage.get_for_send_time() for stage in stage_latency]
    stage_back_send_time_lst = [stage.get_back_send_time() for stage in stage_latency]
    # print(f"exhaustive_partition: {time.time()-s_time:.4f} sec")
    # print(f"partition: {partition}")

    return partition, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst
    
        

from itertools import permutations
def compositions(n, k):
    def inner(n, k):
        if k == 1:
            yield (n,)
        else:
            for i in range(1, n):
                for rest in inner(n-i, k-1):
                    yield (i,) + rest
    return list(inner(n, k))

def ga(num_layer, parallel_config, cost_e, cost_c, gpu_type_lst):
    
    num_layer = num_layer
    pp = parallel_config["pp"]
    num_mb = parallel_config["num_mb"]
    dp_method = parallel_config["dp_method"]
    
    # print(f"cost_e1: {cost_e1}")
    # print(f"cost_e2: {cost_e2}")
    
    if pp.item() > 1:
                                        
        # Configuration 
        num_generations = 50
        num_genes = int(pp.item())
        percent_solutions=0.2
        num_solutions_per_population = 1000
        num_parents_mating = int(num_solutions_per_population * percent_solutions)
        init_range_low=1
        mutation_percent_genes=0.5
        method="std"
        get_plot=False
        save_best_solutions=True
        save_solutions=True
        # parallel_processing="['process',32]"
        # fitness_batch_size=50
        # fitness_batch_size=
        # logger 
        import logging
        level = logging.DEBUG
        
        add_name = datetime.now().strftime('%Y%m%d%H%M%S')
        # _parent_mating_rate_{percent_solutions*100}_
        name = f'./experiment/A100_A10_2/{method}_L_{num_layer}_pp_{int(pp)}_g_{num_generations}_spp_{num_solutions_per_population}_'+ add_name
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        file_handler = logging.FileHandler(name, 'a+', 'utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        console_handler=logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format=logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # initial population
        parent_selection_type="rank"
        crossover_type="single_point"
        mutation_type="swap"
        
        def on_start(ga_instance):
            # print("on_start()")
            # ga_instance.logger.info(f"(mbs, tp, pp, dp) = ({mbs}, {tp}, {dp}, {pp})")
            ga_instance.logger.info(f"gpu type: {gpu_type_lst}")
            ga_instance.logger.info(f"file location: {name}")
            ga_instance.logger.info(f"fitness function method: {method}")
            ga_instance.logger.info(f"num_layers: {num_layer-2}")
            ga_instance.logger.info(f"num_genes: {ga_instance.num_genes}")
            ga_instance.logger.info(f"num_generations: {ga_instance.num_generations}")
            ga_instance.logger.info(f"num_solutions_per_population: {ga_instance.sol_per_pop}")
            ga_instance.logger.info(f"parents_mating: per {ga_instance.num_parents_mating/ga_instance.sol_per_pop*100:.2f}%, num: {ga_instance.num_parents_mating}")
            ga_instance.logger.info(f"percent_mutation: {ga_instance.mutation_percent_genes}%")
            ga_instance.logger.info(f"save_best_solutions: {ga_instance.save_best_solutions}")
            ga_instance.logger.info(f"save_solutions: {ga_instance.save_solutions}")
            
            
        def on_fitness(ga_instance, population_fitness):
            print("on_fitness()")
            # print(f"population_fitness: {population_fitness}")
            # print(f"best_score: {max(population_fitness):0.5f}")

        def on_parents(ga_instance, selected_parents):
            print("on_parents()")
            # print(selected_parents)

        def on_crossover(ga_instance, offspring_crossover):
            print("on_crossover()")
            print(offspring_crossover)
            # print(offspring_crossover.sum(axis=1))

        def on_mutation(ga_instance, offspring_mutation):
            print("on_mutation()")

        def on_generation(ga_instance):
            # print("on_generation()")
            # print(f"Generation = {ga_instance.generations_completed}")
            # print(f"Fitness = {ga_instance.best_solution()[1]}")
            solution = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[0]
            # print(partition)
            # print(partition.sum(axis=0))
            stage_latency = get_stage_latency(solution, cost_e, cost_c, gpu_type_lst)
            # stage_comp_time_lst = [stage.get_comp_time() for stage in stage_latency]
            stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
            _max = max(stage_time_lst)
            _min = min(stage_time_lst)
            max_index = stage_time_lst.index(_max)
            ga_instance.logger.info(f"Generation = {ga_instance.generations_completed}")
            ga_instance.logger.info(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]:.5f}")
            ga_instance.logger.info(f"Solutions  = {solution}")        
            ga_instance.logger.info(f"max - min = {(_max - _min):.8f}, max: {_max:.8f} min: {_min:.8f} max_index: {max_index}")
            
        def on_stop(ga_instance, last_population_fitness):
            # ga_instance.logger.info(f"best solution          = {ga_instance.best_solutions}")
            # ga_instance.logger.info(f"best solutions fitness = {ga_instance.best_solutions_fitness}")
            print("on_stop()")
            # print(f"last_population_fitness: {last_population_fitness}")

        def _fitness_func(ga_instance, solution, solution_idx):
            partition = solution
            # print(partition)
            # print(partition.sum(axis=0))
            stage_latency = get_stage_latency(partition, cost_e, cost_c, gpu_type_lst)
            stage_comp_time_lst = [stage.get_comp_time() for stage in stage_latency]
            stage_for_send_time_lst = [stage.get_for_send_time() for stage in stage_latency]
            stage_back_send_time_lst = [stage.get_back_send_time() for stage in stage_latency]
            dp_method="dp"
            pipecost, _ = schedule(pp, num_mb,stage_comp_time_lst,stage_for_send_time_lst, stage_back_send_time_lst,dp_method)
            fitness = 1/pipecost.item()
            return fitness

        def fitness_func(ga_instance, solution, solution_idx):
            # s = time.time()
            partition = solution
            # print(partition)
            # print(partition.sum(axis=0))
            stage_latency = get_stage_latency(partition, cost_e, cost_c, gpu_type_lst)
            stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
            # stage_comp_time_lst = [stage.get_comp_time() for stage in stage_latency]
            
            # min_latency = min(stage_comp_time_lst)
            # diff = max_latency - min_latency
            # std = np.std(stage_comp_time_lst)
            # f = std.item()
            # e= time.time()
            # ga_instance.logger.info(f"fitness func time: {e-s:.5f}")
            if method == "max":
                max_latency = max(stage_time_lst)
                # ga_instance.logger.info(f"std: {f:.5f}")
                return max_latency
            elif method == "std":
                std = np.std(stage_time_lst)
                # ga_instance.logger.info(f"max: {max_latency:.5f}")
                return std
            elif method == "diff":
                max_latency = max(stage_time_lst)
                min_latency = min(stage_time_lst)
                diff = max_latency - min_latency
                # ga_instance.logger.info(f"max: {max_latency:.5f} min: {min_latency:.5f}")
                # ga_instance.logger.info(f"std: {f:.5f}")
                return diff
        
        def generate_init_population(num_genes, num_layer, init_range_low, num_solutions_per_population):
            # Step 1: Ensure that each element is at least min_value
            pop_size = (num_solutions_per_population, num_genes)
            base_array = np.full(pop_size, init_range_low, dtype=int)
            base_sum = np.sum(base_array[0])
            
            # Step 2: Calculate the remaining sum to distribute
            remaining_sum = num_layer - base_sum
            
            # Step 3: Generate random increments to distribute the remaining sum
            random_values = np.random.randint(0, remaining_sum, (num_solutions_per_population, num_genes - 1))
            random_values.sort(axis=1)
            random_values = np.hstack([np.zeros((num_solutions_per_population, 1), dtype=int), random_values, np.full((num_solutions_per_population, 1), remaining_sum, dtype=int)])
            # random_values = np.concatenate(([0], random_values, [remaining_sum]))
            
            # Step 4: Compute the differences to get the increments
            increments = np.diff(random_values, axis=1)
            
            # Step 5: Add increments to the base array
            result_array = base_array + increments
            # assert False, "Done!"
            # print(result_array)
            return result_array
        
        initial_population=generate_init_population(num_genes, num_layer, init_range_low, num_solutions_per_population)
        
        def crossover_func(parents, offspring_size, ga_instance):
            # This is single-point crossover.
            # s = time.time()
            offspring = []
            idx = 0
            # ga_instance.logger.info(f"parents: {parents}")
            while len(offspring) != offspring_size[0]:
                parent1 = parents[idx % parents.shape[0], :].copy()
                parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
                random_split_point = np.random.choice(range(offspring_size[1]))

                parent1[random_split_point:] = parent2[random_split_point:]

                # print(f"offspring_1: {parent1}")
                if parent1.sum(axis=0) == num_layer:
                    
                    offspring.append(parent1)
                    idx += 1
                    # ga_instance.logger.info(f"offspring: {len(offspring)}")
                    # ga_instance.logger.info(f"idx: {idx}")
                    # ga_instance.logger.info(f"parent1: {parent1}")
                    # ga_instance.logger.info(f"parent2: {parent2}")
                    # ga_instance.logger.info(f"random_split_point: {random_split_point}")
                    # ga_instance.logger.info(f"offspring: {parent1}")
                    
                    # print(f"offspring_2: {parent1}")
                else:
                    pass
            # e = time.time()
            # ga_instance.logger.info(f"crossover time: {e - s:.5f} s")
            return np.array(offspring)
    
        start_time = time.time()
        ga_instance = GA(num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_func,
            # fitness_type='single-point',
            sol_per_pop=num_solutions_per_population,
            num_genes=num_genes,
            gene_type=int,
            initial_population=initial_population,
            parent_selection_type=parent_selection_type,
            crossover_type=crossover_func,
            mutation_type=mutation_type,
            save_best_solutions=save_best_solutions,
            save_solutions=save_solutions,
            mutation_percent_genes=mutation_percent_genes,
            on_start=on_start,
            on_generation=on_generation,
            # parallel_processing=["process",32],
            # fitness_batch_size=fitness_batch_size,
            logger=logger)
            # on_start=on_start,
            # on_fitness=on_fitness,
            # on_parents=on_parents,
            # on_crossover=on_crossover,
            # on_mutation=on_mutation,
            
            # on_stop=on_stop)
        
        ga_instance.run()
        end_time = time.time()
    
        if get_plot:
            ga_instance.plot_fitness(title=f"L{num_layer}_pp{int(pp)}", save_dir=name+".png", plot_type="scatter")
    
        # ga_instance.plot_fitness()
        
        solution, solution_fitness, _ = ga_instance.best_solution()
    
        ga_instance.summary()
    
        ga_instance.logger.info(f"Best solution         : {solution}")
        ga_instance.logger.info(f"Best solution fitness : {solution_fitness:.8f}")
        # ga_instance(f"Index of the best solution: {solution_idx}")
        ga_instance.logger.info(f"ga runtime            : {end_time - start_time:.2f}s")
        logger.handlers.clear()    
        partition = solution

    else:
        partition = [num_layer]

    stage_latency = get_stage_latency(partition, cost_e, cost_c, gpu_type_lst)
    stage_comp_time_lst = [stage.get_comp_time() for stage in stage_latency]
    stage_for_send_time_lst = [stage.get_for_send_time() for stage in stage_latency]
    stage_back_send_time_lst = [stage.get_back_send_time() for stage in stage_latency]
    pipecost, _ = schedule(pp, num_mb,stage_comp_time_lst,stage_for_send_time_lst, stage_back_send_time_lst,dp_method)
    # assert False, "Done!"
    return partition, pipecost, _

def ILP(num_layer, parallel_config, cost_e, cost_c, gpu_type_lst):
    from docplex.mp.model import Model
    
    num_layer = num_layer # 48, input emb, output emb 제외한 개수
    
    pp = int(parallel_config["pp"])
    num_mb = parallel_config["num_mb"]
    dp_method = parallel_config["dp_method"]
    
    # get cplex solver
    m = Model(name='partitioning')
    
    # create variables (#layers for each stage)
    layers_per_stage = []
    for i in range(pp):
        layers_per_stage.append(m.integer_var(lb=1, ub=num_layer-pp, name=f'stage_{i}'))
    
    # print(f"layers_per_stage: {layers_per_stage}")
    
    # print(f"num_layer: {num_layer}")

    print(f"num_layer: {num_layer}")
    # num_layer = 48

    # make a list of cost_e, which is a list of cost_e1 or cost_e2 for each stage
    cost_e_lst =[] # shape: 1 x pp. gpu type 위치마다 cost_e값을 넣어둠
    for i in range(pp):
        if gpu_type_lst[i] == 'A100':
            cost_e_lst.append(cost_e["A100"])
        elif gpu_type_lst[i] == 'A10':
            cost_e_lst.append(cost_e["A10"])
        elif gpu_type_lst[i] == 'A6000':
            cost_e_lst.append(cost_e["A6000"])
        else:
            assert False, "gpu type is not recognized"
    # print(f"cost_e_lst: {cost_e_lst}")
    
    # calculate embedding and communication time for each stage, which is a constant
    # embedding_and_comm_time = [0 for i in range(num_layer)]
    embedding_and_comm_time = []
    # transformer_layer_time = []
    # print(f"cost_c: {len(cost_c[0])} {cost_c[0]}")
    # cost_e_lst pp x L
    for j in range(pp):
        if j == 0:
            embedding_and_comm_time.append(cost_e_lst[0][0].item()) # input embedding
            embedding_and_comm_time[j] += cost_c[0][j].item() # for_send_time
        elif j == pp-1:
            embedding_and_comm_time.append(cost_e_lst[pp-1][-1].item()) # output embedding
            embedding_and_comm_time[j] += cost_c[0][j-1].item() # back_send_time
        else:
            # embedding_and_comm_time.append(0)
            embedding_and_comm_time.append(cost_c[0][j-1].item()) # comm_time
        
        # embedding_and_comm_time[i] += cost_c[0][i]
        # if i != 0 and i != pp_degree-1:
            # embedding_and_comm_time[i] += cost_c[0][i-1]
    # cost_c가 모든 레이어에서 동일하다면?
    # print(f"cost_c: {cost_c}")
    # print(f"transformer_layer_time: {transformer_layer_time}")
    # print(f"embedding_and_comm_time: {embedding_and_comm_time}")
    # transformer_layer_time = cost_e_lst[i][1].item()
    # add constraints
    m.add_constraint(m.sum(layers_per_stage) == num_layer)
    m.minimize(m.max( ( layers_per_stage[p] * cost_e_lst[p][1].item() + embedding_and_comm_time[p]) for p in range(pp)))
    # If you'd like to minimize pipeline latency not the max latency, use the following line instead of the above line.
    # m.minimize((num_mb-1)*m.max(layers_per_stage[i]*cost_e[i][1]+embedding_and_comm_time[i] \
    #             for i in range(pp)) + \
    #             m.sum(layers_per_stage[i]*cost_e[i][1]+embedding_and_comm_time[i] \
    #             for i in range(pp)))
    # m.print_information()


    
    solution = m.solve()
    # print(f"solution: {solution}")
    
    partition=[]
    for i in range(pp):
        partition.append(int(solution.get_value(layers_per_stage[i])))
        if i == 0 or i == pp-1:
            partition[i] += 1
    print(f"sum layers: {sum(partition)}")
    '''
    while m.solve():
        s = m.solution
        partition = []
        for i in range(pp):
            partition.append(int(s.get_value(layers_per_stage[i])))
            if i == 0 or i == pp-1:
                partition[i] += 1
                
        print(f"ILP solution: {partition}")
        # m.add_constraint(m.sum(layers_per_stage) == num_layer)
        m.add_constraint(m.max(layers_per_stage[i] * cost_e_lst[i][1].item() + embedding_and_comm_time[i] for i in range(pp)) <= 0.99 * m.max(int(s.get_value(layers_per_stage[i])) * cost_e_lst[i][1].item() + embedding_and_comm_time[i] for i in range(pp)))
        # m.add_constraint(m.sum(layers_per_stage[i]*cost_e[i][1]+embedding_and_comm_time[i] for i in range(pp)) \
        #                 <= 0.99 * m.sum(int(s.get_value(layers_per_stage[i]))*cost_e[i][1]+embedding_and_comm_time[i] for i in range(pp)))
    '''
    
    stage_latency = get_stage_latency(partition, cost_e, cost_c, gpu_type_lst)
    stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
    stage_comp_time_lst = [stage.get_comp_time() for stage in stage_latency]
    stage_for_send_time_lst = [stage.get_for_send_time() for stage in stage_latency]
    stage_back_send_time_lst = [stage.get_back_send_time() for stage in stage_latency]
    pipecost, _ = schedule(pp, num_mb, stage_comp_time_lst, stage_for_send_time_lst, stage_back_send_time_lst, dp_method)
    print(f"max_latency: {max(stage_time_lst)*1000:.5f}")
    
    return partition, stage_comp_time_lst, stage_for_send_time_lst, stage_back_send_time_lst