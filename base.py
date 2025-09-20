from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class TSPAlgorithm(ABC):
    """TSP算法抽象基类"""

    def __init__(self, dist_mat: np.ndarray, config):
        """初始化算法

        Args:
            dist_mat: 距离矩阵
            config: 配置参数
        """
        self.dist_mat = dist_mat
        self.config = config
        self.n_cities = dist_mat.shape[0]

        # 算法状态
        self.best_path = None
        self.best_distance = float('inf')
        self.history = []  # 存储每代最佳路径和距离

    @abstractmethod
    def run(self) -> Tuple[List[List[int]], List[float]]:
        """运行算法

        Returns:
            Tuple[List[List[int]], List[float]]:
                每代最佳路径列表和每代最佳距离列表
        """
        pass

    def get_result(self) -> Tuple[List[int], float]:
        """获取最终结果

        Returns:
            Tuple[List[int], float]: 最佳路径和最佳距离
        """
        return self.best_path, self.best_distance