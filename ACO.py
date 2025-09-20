import numpy as np
import random
from typing import List, Tuple
from base import TSPAlgorithm


class AntColonyOptimization(TSPAlgorithm):
    """蚁群算法实现"""

    def __init__(self, dist_mat: np.ndarray, config):
        """初始化蚁群算法

        Args:
            dist_mat: 城市距离矩阵
            config: 配置参数对象
        """
        super().__init__(dist_mat, config)

        # 初始化信息素矩阵
        self.pheromone = np.ones((self.n_cities, self.n_cities)) * 0.1

        # 初始化启发式信息矩阵（距离的倒数）
        self.heuristic = 1 / (dist_mat + np.eye(self.n_cities) * 1e-10)  # 避免除以零

        # 蚂蚁数量
        self.n_ants = config.aco_ant_num

    def _initialize_ants(self):
        """初始化蚂蚁"""
        self.ants = []
        for _ in range(self.n_ants):
            # 每只蚂蚁从随机城市开始（但确保至少一只从城市0开始）
            start_city = 0 if len(self.ants) == 0 else random.randint(0, self.n_cities - 1)
            ant = {
                'path': [start_city],
                'visited': {start_city},
                'distance': 0.0
            }
            self.ants.append(ant)

    def _select_next_city(self, ant):
        """选择下一个城市

        Args:
            ant: 蚂蚁当前状态

        Returns:
            int: 下一个城市的索引
        """
        current_city = ant['path'][-1]
        unvisited = [city for city in range(self.n_cities) if city not in ant['visited']]

        if not unvisited:
            return None

        # 计算选择概率
        probabilities = []
        for city in unvisited:
            pheromone = self.pheromone[current_city, city] ** self.config.aco_alpha
            heuristic = self.heuristic[current_city, city] ** self.config.aco_beta
            probabilities.append(pheromone * heuristic)

        # 归一化概率
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()

        # 根据概率选择下一个城市
        return np.random.choice(unvisited, p=probabilities)

    def _construct_solutions(self):
        """构建解"""
        for ant in self.ants:
            # 继续构建路径直到访问所有城市
            while len(ant['path']) < self.n_cities:
                next_city = self._select_next_city(ant)
                if next_city is None:
                    break

                current_city = ant['path'][-1]
                ant['path'].append(next_city)
                ant['visited'].add(next_city)
                ant['distance'] += self.dist_mat[current_city, next_city]

            # 闭合路径（返回起点）
            if len(ant['path']) == self.n_cities:
                first_city = ant['path'][0]
                last_city = ant['path'][-1]
                ant['distance'] += self.dist_mat[last_city, first_city]

    def _update_pheromone(self):
        """更新信息素"""
        # 信息素挥发
        self.pheromone *= (1 - self.config.aco_rho)

        # 信息素增加
        for ant in self.ants:
            path = ant['path']
            path_length = len(path)

            # 计算信息素增量
            if self.config.aco_strategy == 0:  # 蚂蚁周期系统
                delta = self.config.aco_q / ant['distance'] if ant['distance'] > 0 else 0
            elif self.config.aco_strategy == 1:  # 蚂蚁数量系统
                delta = self.config.aco_q / ant['distance'] if ant['distance'] > 0 else 0
            else:  # 蚂蚁密度系统
                delta = self.config.aco_q

            # 更新路径上的信息素
            for i in range(path_length - 1):
                city_i = path[i]
                city_j = path[i + 1]
                self.pheromone[city_i, city_j] += delta
                self.pheromone[city_j, city_i] += delta  # 对称矩阵

            # 闭合路径的信息素更新
            if path_length > 1:
                self.pheromone[path[-1], path[0]] += delta
                self.pheromone[path[0], path[-1]] += delta

    def _update_best_solution(self):
        """更新最佳解"""
        for ant in self.ants:
            if ant['distance'] < self.best_distance:
                self.best_distance = ant['distance']
                self.best_path = ant['path'].copy()

    def run(self) -> Tuple[List[List[int]], List[float]]:
        """运行蚁群算法

        Returns:
            Tuple[List[List[int]], List[float]]:
                每代最佳路径列表和每代最佳距离列表
        """
        # 迭代过程
        for gen in range(self.config.aco_gen_num):
            # 初始化蚂蚁
            self._initialize_ants()

            # 构建解
            self._construct_solutions()

            # 更新信息素
            self._update_pheromone()

            # 更新最佳解
            self._update_best_solution()

            # 记录当前代的最佳结果
            self.history.append({
                'path': self.best_path.copy(),
                'distance': self.best_distance
            })

            # 每50代打印进度
            if (gen + 1) % 50 == 0:
                print(f"迭代 {gen + 1}/{self.config.aco_gen_num}, 最佳距离: {self.best_distance:.2f}")

        # 提取历史记录
        result_list = [h['path'] for h in self.history]
        fitness_list = [h['distance'] for h in self.history]

        return result_list, fitness_list