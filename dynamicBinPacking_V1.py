import random
from deap import base, creator, tools, algorithms
import numpy as np
import timeit
from pathos.multiprocessing import ProcessingPool as Pool
from matplotlib import pyplot as plt
import time
import datetime
from itertools import repeat
from scoop import futures

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

# 物品数量和物品大小
import random

ITEMS = []
for _ in range(100):
    enter_time = random.randint(1, 100)
    exit_time = random.randint(enter_time + 1, 200)
    item_size = random.randint(10, 100)

    # 确保进入时间小于离开时间
    while enter_time >= exit_time:
        enter_time = random.randint(1, 100)
        exit_time = random.randint(enter_time + 1, 200)

    ITEMS.append((enter_time, exit_time, item_size))

print(ITEMS)
# 容器容量
bin_capacity = 100

item_sizes = [ITEMS[i][2] for i in range(len(ITEMS))]

# 初始化函数
def binpacking_init(item_sizes, bin_cap):
    '''
        对每个个体初始化，采用首次适配原则初始化，为了保证样本随机性，初始化个提示，对所有物品重排序
    '''
    item_num = len(item_sizes)

    rand_idx = np.linspace(0, item_num - 1, item_num).astype(int)
    np.random.shuffle(rand_idx)

    init_ind = np.zeros(item_num).astype(int)
    bins_left_size = np.ones(item_num - 1) * bin_cap  # 最差情况每个箱子放一个物品，初始化所有箱子剩余大小为bin_cap
    bin_idx = 0  # 第一个可用箱子的索引
    for i in range(len(item_sizes)):
        # FIXME:假设所有物品都能放入箱子，所以没有做异常处理，如物品未放入箱子时，如何做
        for bin_idx in range(0, len(bins_left_size)):
            # 找到第一个能放下物品的箱子
            if item_sizes[rand_idx[i]] <= bins_left_size[bin_idx]:
                init_ind[rand_idx[i]] = bin_idx
                bins_left_size[bin_idx] -= item_sizes[rand_idx[i]]
                break
    return init_ind.tolist()

def bin_count(individual):
    box_mapping = {}
    new_result = []

    # 遍历结果列表中的每个值
    for value in individual:
        # 如果值尚未分配新编号，则将其添加到字典中，并将当前字典长度作为新编号
        if value not in box_mapping:
            box_mapping[value] = len(box_mapping)
        # 将新编号添加到新结果列表中
        new_result.append(box_mapping[value])
    bin_num = max(new_result) + 1
    return bin_num

def binpacking_fitness(individual, item_sizes, bin_cap):
    '''
        在初始化、交叉和变异操作时，应保证解的可行性，因此，评估函数我们用使用的箱子个数和空闲空间大小来表示
    '''
    # fitness = alpha * bin_num + beta * idle_size
    # FIXME:超参数，当前写在函数实现里，以后为了代码整体性，需要进行整合
    alpha = 0.7
    beta = 1 - alpha

    # 获取一共用了多少个箱子
    # this_bin_cap = bin_cap
    bin_num = bin_count(individual)
    fitness = bin_num
    return fitness,

def map_individual(individual):
    box_mapping = {}
    new_individual = individual

    # 遍历个体列表中的每个值
    for idx, value in enumerate(individual):
        # 如果值尚未分配新编号，则将其添加到字典中，并将当前字典长度作为新编号
        if value not in box_mapping:
            box_mapping[value] = len(box_mapping)
        # 将新编号添加到新结果列表中
        new_individual[idx] = box_mapping[value]
    return new_individual


def check_bin(bin, ITEMS, bin_cap) -> bool:
    bin_list = list(bin[0])
    selected_items = sorted((ITEMS[i] for i in bin_list), key=lambda x: x[0])  # 获取选定的物品，并按开始时间排序

    for i in range(0, len(selected_items)):
        cur_bin_total_weight = selected_items[i][2]
        for j in range(i + 1, len(selected_items)):
            if selected_items[i][1] > selected_items[j][0]:
                # 对于当前时间点重合的元素，相加看是否大于cap
                cur_bin_total_weight += selected_items[j][2]
                if cur_bin_total_weight > bin_cap:
                    return False
    return True

def curtime_cap_check(individual, ITEMS, bin_cap):
    # this_individual = individual
    individual = map_individual(individual) # 映射individual

    #根据物品的进入时间对 individual 中的物品进行排序
    sorted_items = sorted(enumerate(individual), key=lambda x: ITEMS[x[1]][0])
    bins = [[] for _ in range(max(individual) + 1)]

    for i, bin_idx in sorted_items:
         bins[bin_idx].append(i) # bin中的物品按进入时间排序

    for bin_idx, bin in enumerate(bins):
        if not check_bin([bin], ITEMS, bin_cap):
            sorted_bin = sorted(bin, key=lambda x: ITEMS[x][0], reverse=True)

            for item_idx in sorted_bin:

                # 创建一个列表，用于存储所有可以容纳当前物品的箱子
                possible_bins = []
                # 对每个箱子进行检查
                for other_bin_idx, other_bin in enumerate(bins):
                    if other_bin_idx != bin_idx:
                        # 尝试将物品添加到当前箱子
                        other_bin.append(item_idx)

                        # 添加后检查箱子是否符合约束
                        if check_bin([other_bin], ITEMS, bin_cap):
                            # 如果箱子符合约束，就将箱子添加到可能的箱子列表中
                            possible_bins.append(other_bin_idx)

                        # 将物品从箱子中移除
                        other_bin.remove(item_idx)

                # 如果有可以容纳当前物品的箱子
                if possible_bins:
                    # 选择一个箱子，优先选择物品数量最少的箱子
                    selected_bin_idx = min(possible_bins, key=lambda x: len(bins[x]))

                    # 将物品移动到选择的箱子
                    bins[selected_bin_idx].append(item_idx)
                    bins[bin_idx].remove(item_idx)

                else:
                    # 如果没有箱子可以容纳当前物品，就新建一个箱子
                    bins.append([item_idx])
                    bins[bin_idx].remove(item_idx)

                if check_bin([bin], ITEMS, bin_cap):  # 不需要再移除了:
                    break

    for bin_idx, bin in enumerate(bins):
        for item_idx in bin:
            individual[item_idx] = bin_idx

    return individual

def cxTwoPointCopy(ind1, ind2):
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

    return ind1, ind2

def binpacking_cxTwoPoint(ind1, ind2, ITEMS, bin_cap):
    '''
        交叉算子-两点交叉
    '''
    ind1, ind2 = cxTwoPointCopy(ind1, ind2)
    ind1 = curtime_cap_check(ind1, ITEMS, bin_cap)
    ind2 = curtime_cap_check(ind2, ITEMS, bin_cap)
    return ind1, ind2

def binpacking_mutUniformInt(individual, indpb, item_sizes, ITEMS, bin_cap):
    '''
        变异算子
    '''
    tools.mutUniformInt(individual, low=min(item_sizes), up=max(item_sizes), indpb=indpb)
    this_individual = curtime_cap_check(individual, ITEMS, bin_cap)
    return this_individual,

def binpacking_selTournament(individuals, k, tournsize, fit_attr="fitness"):
    '''
        选择算子
    '''
    return tools.selTournament(individuals, k, tournsize, fit_attr="fitness")


def bins_state(info, item_sizes, ind, stop=False):
    ind = np.array(ind)
    item_sizes = np.array(item_sizes)
    print("*************************************")
    print("             ", info, "             ")
    print("*************************************")
    print("Each item size :", item_sizes)
    print("Best individual:", ind)
    print("Each bin state:")
    for i in range(0, max(ind) + 1):
        print("Bin ", i, ": ", item_sizes[ind == i])

    if stop:
        a = input()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 注册
toolbox.register("initialize", binpacking_init, item_sizes=item_sizes, bin_cap=bin_capacity)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.initialize)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", binpacking_fitness, item_sizes=item_sizes, bin_cap=bin_capacity)
toolbox.register("mate", binpacking_cxTwoPoint, ITEMS=ITEMS, bin_cap=bin_capacity)
toolbox.register("mutate", binpacking_mutUniformInt, indpb=0.5, item_sizes=item_sizes, ITEMS = ITEMS, bin_cap=bin_capacity)
toolbox.register("select", binpacking_selTournament, tournsize=3)


pop_size = 100
num_generations = 1000
cxpb = 0.5
mutpb = 0.5

# 初始化种群

pop = toolbox.population(n=pop_size)
#
# fitnessValuesMean_arr = []
# fitnessValuesMax_arr = []
#
# # 运行遗传算法
# for gen in range(num_generations):
#     # 选择算子
#     # 选择这一代
#     offspring = tools.selBest(pop, pop_size)
#     fitnessValues = list(map(toolbox.evaluate, offspring))
#
#     fitnessValuesMax_arr.append(np.min(fitnessValues))
#     fitnessValuesMean_arr.append(np.mean(fitnessValues))
#
#     if (gen == 0) or ((gen + 1) % 100 == 0):
#         print(datetime.datetime.now(), "第 %s 轮进化(本代大小/种群大小：%d/%d)：最优适应度 %d | 平均适应度 %d " % (
#         gen, len(offspring), len(pop), np.min(fitnessValues), np.mean(fitnessValues)))
#
#     freshIndividuals = []
#     # 交叉操作
#     if gen == 899:
#         print("debug>>>>>>")
#     for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
#         if random.random() < cxpb:
#             child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
#             toolbox.mate(child1, child2)
#             freshIndividuals.append(child1)
#             freshIndividuals.append(child2)
#             del child1.fitness.values
#             del child2.fitness.values
#
#     # 突变操作
#     for mutant in offspring:
#         if random.random() < mutpb:
#             child = toolbox.clone(mutant)
#             toolbox.mutate(child)
#             freshIndividuals.append(child)
#             del child.fitness.values
#
#     # 评估新生成的个体
#     for ind in freshIndividuals:
#         if not ind.fitness.valid:
#             ind.fitness.values = toolbox.evaluate(ind)
#
#     # 更新种群
#     pop += freshIndividuals
#     pop = tools.selBest(pop, pop_size * 2)
#
# # 输出结果
# best_ind = tools.selBest(pop, 1)[0]
# print(best_ind.fitness.values)
# Best_fitness = best_ind.fitness.values
#
# print("Best fitness: ", Best_fitness)
# bins_state("Bin state", item_sizes, best_ind)
#
# plt.figure()
# plt.title("DynamicBinPackingV1")
# plt.plot(fitnessValuesMax_arr, label="Best")
# plt.plot(fitnessValuesMean_arr, label="Mean")
# plt.legend(["best", "mean"])
# plt.show()
#
# exit()

if __name__ == '__main__':
    # 并行计算Fitness，目前用不到，速度反而显著降低
    # pool = Pool(64)
    # toolbox.register("map", pool.map)
    # toolbox.register("map", futures.map)
    # 添加时间戳
    start_time = timeit.default_timer()
    # 运行遗传算法
    hof = tools.HallOfFame(1)  # maxSize
    resultPop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.5, ngen=num_generations, halloffame=hof)
    #获得最优
    print(hof[0].fitness.values)
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print(f"Executed the script in {execution_time} seconds")

    best_ind = tools.selBest(pop, 1)[0]
    Best_fitness = best_ind.fitness.values
    print("Best fitness: ", Best_fitness)
    bins_state("Bin state", item_sizes, best_ind)


'''
考虑物品的属性、箱子容量的属性（多维） 对应多种密码算法类型
DAG 
先不考虑依赖关系，只考虑多维下的装箱
DAG： 加上带宽通信约束 NSGA-II
并行
'''