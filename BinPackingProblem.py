import random
from deap import base, creator, tools

'''
对于n个物品，求能装下该物品的最小bin数目，先令len(bin) = n，再逐步优化
首先初始化种群中的个体为：("individual", tools.initRepeat, creator.Individual, toolbox.random_bin, n=num_items)，表示将物品随机放入一个箱子，individual的index为物品编号，value为放入的箱子
种群就是pop_size = 100，为100个个体
遗传的代数为1000
经过交叉、突变操作重新生成新个体
通过selBest函数选择最符合要求的个体
'''
# 物品数量和物品大小
num_items = 10
item_sizes = [1, 5, 5, 6, 3, 2, 4, 7, 9, 8]

# 容器容量
bin_capacity = 10

# 创建适应度函数和个体
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 初始化个体
toolbox.register("random_bin", random.randint, 0, 5)  #0 ~ n - 1 n个箱子编号随机生成
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.random_bin, n=num_items)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# 评估函数
def binpacking_fitness(individual):
    bins = {}
    for idx, bin_idx in enumerate(individual):
        if bin_idx not in bins:
            bins[bin_idx] = 0
        bins[bin_idx] += item_sizes[idx]
    #over_capacity / bin_capacity。这个惩罚项的作用是，当某个箱子超过其容量时，增加适应度值，使解变得更糟糕。
    over_capacity = sum(max(0, bins[i] - bin_capacity) for i in bins)
    num_bins = len(bins)
    return num_bins + (over_capacity / bin_capacity), #逗号表示返回的是一个具有单个元素的元组，而不仅仅是一个数值。



toolbox.register("evaluate", binpacking_fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=num_items - 1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# 遗传算法参数
pop_size = 100
num_generations = 1000
cxpb = 0.5
mutpb = 0.5

# 初始化种群
pop = toolbox.population(n=pop_size)

offspring = toolbox.select(pop, len(pop))
offspring = list(offspring)

for ind in offspring:
    if not ind.fitness.valid:
        ind.fitness.values = toolbox.evaluate(ind)

# 运行遗传算法
for gen in range(num_generations):
    offspring = toolbox.select(pop, len(pop))
    offspring = list(offspring)

    # 交叉操作
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # 突变操作
    for mutant in offspring:
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # 评估新生成的个体
    for ind in offspring:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)

    # 更新种群
    pop[:] = offspring


# 输出结果
best_ind = tools.selBest(pop, 1)[0]
print("Best individual:", best_ind)
print("Best fitness:", best_ind.fitness.values[0])


# 创建一个空字典来存储重新编号的箱子
box_mapping = {}
new_result = []

# 遍历结果列表中的每个值
for value in best_ind:
    # 如果值尚未分配新编号，则将其添加到字典中，并将当前字典长度减 1 作为新编号
    if value not in box_mapping:
        box_mapping[value] = len(box_mapping)

    # 将新编号添加到新结果列表中
    new_result.append(box_mapping[value])

print(new_result)
print(max(new_result) + 1)