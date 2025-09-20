import numpy as np
import random
from typing import List, Tuple
from base import TSPAlgorithm


class Individual:
    """个体类，表示一个可能的解（城市访问顺序）"""

    def __init__(self, genes: List[int] = None, dist_mat: np.ndarray = None):
        """初始化个体

        Args:
            genes: 基因序列（城市访问顺序），如果为None则随机生成
            dist_mat: 城市距离矩阵
        """
        self.dist_mat = dist_mat
        self.gene_len = dist_mat.shape[0] if dist_mat is not None else 0

        # 随机生成基因序列（如果未提供）
        if genes is None:
            # 确保从城市0开始
            genes = [0]  # 起始城市为0
            # 随机排列剩余城市
            remaining_cities = list(range(1, self.gene_len))
            random.shuffle(remaining_cities)
            genes.extend(remaining_cities)

        self.genes = genes
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self) -> float:
        """计算个体适应度（路径总长度）

        Returns:
            float: 适应度值（路径总长度）
        """
        if self.dist_mat is None:
            return float('inf')

        # 使用numpy向量化计算提高效率
        genes_shifted = np.roll(self.genes, -1)  # 将序列向左移动一位
        fitness = np.sum(self.dist_mat[self.genes, genes_shifted])
        return fitness

    def __lt__(self, other):
        """定义小于运算符，用于排序"""
        return self.fitness < other.fitness


class GeneticAlgorithm(TSPAlgorithm):
    """遗传算法实现"""

    def __init__(self, dist_mat: np.ndarray, config):
        """初始化遗传算法

        Args:
            dist_mat: 城市距离矩阵
            config: 配置参数对象
        """
        super().__init__(dist_mat, config)

    def _initialize_population(self):
        """初始化种群"""
        self.individuals = [Individual(dist_mat=self.dist_mat)
                            for _ in range(self.config.ga_individual_num)]
        self.best_individual = min(self.individuals)

        # 更新全局最佳
        if self.best_individual.fitness < self.best_distance:
            self.best_path = self.best_individual.genes.copy()
            self.best_distance = self.best_individual.fitness

    def _selection(self) -> List[Individual]:
        """锦标赛选择

        Returns:
            List[Individual]: 被选中的个体列表
        """
        selected = []
        for _ in range(self.config.ga_individual_num):
            # 随机选择一组个体进行锦标赛
            tournament = random.sample(self.individuals, self.config.ga_tournament_size)
            # 选择适应度最好的个体
            winner = min(tournament)
            selected.append(Individual(genes=winner.genes.copy(), dist_mat=self.dist_mat))
        return selected

    def _ordered_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """顺序交叉(OX)操作

        Args:
            parent1: 父代个体1
            parent2: 父代个体2

        Returns:
            Tuple[Individual, Individual]: 两个子代个体
        """
        # 随机选择交叉点
        point1 = random.randint(0, self.n_cities - 2)
        point2 = random.randint(point1 + 1, self.n_cities - 1)

        # 创建子代基因
        child1_genes = [-1] * self.n_cities
        child2_genes = [-1] * self.n_cities

        # 复制中间段
        child1_genes[point1:point2] = parent1.genes[point1:point2]
        child2_genes[point1:point2] = parent2.genes[point1:point2]

        # 填充剩余位置
        self._fill_genes(child1_genes, parent2.genes, point1, point2)
        self._fill_genes(child2_genes, parent1.genes, point1, point2)

        # 创建子代个体
        child1 = Individual(genes=child1_genes, dist_mat=self.dist_mat)
        child2 = Individual(genes=child2_genes, dist_mat=self.dist_mat)

        return child1, child2

    def _fill_genes(self, child_genes: List[int], parent_genes: List[int],
                    point1: int, point2: int):
        """填充子代基因的剩余位置

        Args:
            child_genes: 子代基因序列
            parent_genes: 父代基因序列
            point1: 交叉起始点
            point2: 交叉结束点
        """
        # 找出父代中不在子代中间段的基因
        remaining_genes = [gene for gene in parent_genes if gene not in child_genes[point1:point2]]

        # 计算需要填充的位置
        fill_indices = list(range(point2, self.n_cities)) + list(range(0, point1))

        # 填充基因
        for i, idx in enumerate(fill_indices):
            if i < len(remaining_genes):
                child_genes[idx] = remaining_genes[i]

    def _mutation(self, individual: Individual):
        """交换变异操作

        Args:
            individual: 需要变异的个体
        """
        if random.random() < self.config.ga_mutate_prob:
            # 随机选择两个不同的位置
            idx1, idx2 = random.sample(range(self.n_cities), 2)
            # 交换基因
            individual.genes[idx1], individual.genes[idx2] = individual.genes[idx2], individual.genes[idx1]
            # 更新适应度
            individual.fitness = individual.evaluate_fitness()

    def _evolve(self):
        """执行一代进化"""
        # 选择
        selected = self._selection()

        # 交叉
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            if random.random() < self.config.ga_cross_prob:
                child1, child2 = self._ordered_crossover(selected[i], selected[i + 1])
                offspring.extend([child1, child2])
            else:
                offspring.extend([selected[i], selected[i + 1]])

        # 变异
        for individual in offspring:
            self._mutation(individual)

        # 更新种群
        self.individuals = offspring

        # 更新最佳个体
        current_best = min(self.individuals)
        if current_best.fitness < self.best_individual.fitness:
            self.best_individual = current_best

        # 更新全局最佳
        if self.best_individual.fitness < self.best_distance:
            self.best_path = self.best_individual.genes.copy()
            self.best_distance = self.best_individual.fitness

    def run(self) -> Tuple[List[List[int]], List[float]]:
        """运行遗传算法

        Returns:
            Tuple[List[List[int]], List[float]]:
                每代最佳路径列表和每代最佳距离列表
        """
        # 初始化种群
        self._initialize_population()

        # 进化过程
        for gen in range(self.config.ga_gen_num):
            self._evolve()

            # 记录当前代的最佳结果
            self.history.append({
                'path': self.best_individual.genes.copy(),
                'distance': self.best_individual.fitness
            })

            # 每100代打印进度
            if (gen + 1) % 50 == 0:
                print(f"代 {gen + 1}/{self.config.ga_gen_num}, 最佳距离: {self.best_distance:.2f}")

        # 提取历史记录
        result_list = [h['path'] for h in self.history]
        fitness_list = [h['distance'] for h in self.history]

        return result_list, fitness_list