import random
from deap import base, creator, tools
import numpy as np

from itertools import repeat

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

# 物品数量和物品大小
num_items = 10
item_sizes = [1, 5, 5, 6, 3, 2, 4, 7, 9, 8]

# 容器容量
bin_capacity = 10


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


# 评估函数
def binpacking_fitness(individual, item_sizes, bin_cap):
    '''
        在初始化、交叉和变异操作时，应保证解的可行性，因此，评估函数我们用使用的箱子个数和空闲空间大小来表示
    '''
    # fitness = alpha * bin_num + beta * idle_size
    # FIXME:超参数，当前写在函数实现里，以后为了代码整体性，需要进行整合
    alpha = 0.7
    beta = 1 - alpha

    # 获取一共用了多少个箱子
    bin_num = np.max(individual)
    # 计算空闲空间大小
    idle_size = 0
    for bin_idx in range(bin_num + 1):
        idle_size += (bin_cap - item_sizes[individual == bin_idx])

    fitness = alpha * bin_num + beta * idle_size

    return fitness,


def binpacking_constraint_check(individual, item_sizes, bin_cap):
    '''
        约束检查和调整，当个体不满足约束时，按照先放大物品，后放小物品的基本规则，调整物品放置状态
        但为了保证随机性，在取放物品时，按照物品和箱子剩余空间大小的比例的负相关作为概率进行处理，如：
        取物品时：箱子大小为10，里面有4个物品，大小分别是3，3，4，5，则按照权重15/5，15/5，15/5，15/5
        的概率循环取物品，权重越高，取到物品的概率越大。
        放物品时：待放入物品大小为3，共有4个可用箱子，可用空间分别为2，3，4，5，则按照0，12/3，12/4
        12/5的概率放入物品，权重越高，放入的概率越大。
    '''
    '''
        FIXME:当前实现是按顺序取出一个或多个物品直至使其满足约束，并放入到能容纳该物品的第一个箱子，上述功能后续再实现
    '''
    # 检查个体是否合规
    bin_num = max(individual) + 1
    #individual = np.array(individual)
    item_sizes = np.array(item_sizes)
    for bin_idx in range(bin_num):  # 遍历所有箱子
        item_in_bin = item_sizes[individual == bin_idx]  # 计算箱子bin_idx中所有物品
        sum_size_in_bin = sum(item_in_bin)
        if sum_size_in_bin <= bin_cap:  # 如果箱子中物品大小之和不超过箱子大小，则检查下一个箱子
            continue
        # 对于超过箱子大小的物品，按照首次适配原则拿出和重新放置
        item_idxs = np.where(individual == bin_idx)[0]  # 记录所有放在箱子bin_idx中的物品索引
        for item_idx in range(len(item_in_bin)):  # 遍历所有放在箱子bin_idx中的物品
            satisfied = False  # 对于需要拿出的物品，当前箱子是否满足，
            for bin_idx2 in range(bin_num):
                if sum(item_sizes[individual == bin_idx2]) + item_sizes[item_idxs[item_idx]] <= bin_cap:
                    individual[item_idxs[item_idx]] = bin_idx2  # 更新物品放置的箱子，即更新个体编码
                    satisfied = True
                    break
            # 如果当前箱子满足不了要求，就开一个新箱子
            if not satisfied:
                individual[item_idxs[item_idx]] = bin_num
                bin_num += 1
            sum_size_in_bin -= item_in_bin[item_idx]  # 更新物品中所有物品的总大小

            if sum_size_in_bin <= bin_cap:  # 直到满足约束条件
                break
    return individual
def binpacking_constraint_check_2(individual, item_sizes, bin_cap):
    bin_loads = [0] * (max(individual) + 1)

    for item, bin_idx in enumerate(individual): #bin_idx: 箱子下标
        bin_loads[bin_idx] += item_sizes[item]

    for bin_idx, load in enumerate(bin_loads): #对于每一个箱子
        if load > bin_cap: #如果溢出
            # 将溢出的箱子中的物品移到新的箱子中
            overfilled_items = [i for i, b in enumerate(individual) if b == bin_idx]  #找到溢出的物品的下标
            individual[overfilled_items.pop()] = len(bin_loads)  # 创建一个新的箱子

            while overfilled_items:
                item_to_move = overfilled_items.pop()
                found_bin = False
                for b_idx, load in enumerate(bin_loads):
                    if load + item_sizes[item_to_move] <= bin_cap:
                        individual[item_to_move] = b_idx
                        bin_loads[b_idx] += item_sizes[item_to_move]
                        found_bin = True
                        break

                if not found_bin:
                    individual[item_to_move] = len(bin_loads)
                    bin_loads.append(item_sizes[item_to_move])

    return individual


def binpacking_cxTwoPoint(ind1, ind2, item_sizes, bin_cap):
    '''
        交叉算子-两点交叉
    '''
    ind1, ind2 = tools.cxTwoPoint(ind1, ind2)
    ind1 = binpacking_constraint_check(ind1, item_sizes, bin_cap)
    ind2 = binpacking_constraint_check(ind2, item_sizes, bin_cap)

    return ind1, ind2


def binpacking_mutUniformInt(individual, indpb, item_sizes, bin_cap):
    '''
        变异算子
    '''
    #his_individual = tools.mutUniformInt(individual, low=min(item_sizes), up=max(item_sizes), indpb= 1)
    low = min(item_sizes)
    up = max(item_sizes)
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(range(size), low, up):
        if random.random() < indpb:
            individual[i] = random.randint(xl, xu)

    this_individual = binpacking_constraint_check(individual, item_sizes, bin_cap)
    bins_state("变异算子", item_sizes, individual)
    return this_individual


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


# 创建适应度函数和个体
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 初始化个体
toolbox.register("initalize", binpacking_init, item_sizes=item_sizes, bin_cap=bin_capacity)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.initalize)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", binpacking_fitness, item_sizes=item_sizes, bin_cap=bin_capacity)
toolbox.register("mate", binpacking_cxTwoPoint, item_sizes=item_sizes, bin_cap=bin_capacity)
toolbox.register("mutate", binpacking_mutUniformInt, indpb=0.1, item_sizes=item_sizes, bin_cap=bin_capacity)
toolbox.register("select", binpacking_selTournament, tournsize=3)

# 遗传算法参数
pop_size = 100
num_generations = 50
cxpb = 0.5
mutpb = 0.5

# 初始化种群
pop = toolbox.population(n=pop_size)
for i, ind in enumerate(pop):
    print(f"Individual {i + 1}: fitness.valid = {ind.fitness.valid}")
# 运行遗传算法
for gen in range(num_generations):
    offspring = toolbox.select(pop, len(pop))
    offspring = list(offspring)
    #print("第 %s 轮进化：" %(gen))
    # 交叉操作
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # 突变操作
    for mutant in offspring:
        if random.random() < mutpb:
            bins_state("变异前：", item_sizes, mutant)
            toolbox.mutate(mutant)
            bins_state("变异后：", item_sizes, mutant)
            del mutant.fitness.values

    # 评估新生成的个体
    for ind in offspring:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)

    # 更新种群
    pop[:] = offspring

# 输出结果
best_ind = tools.selBest(pop, 1)[0]
Best_fitness = best_ind.fitness.values[0]

print("Best fitness: ", Best_fitness)
bins_state("Bin state", item_sizes, best_ind)
