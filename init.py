from GA import GeneticAlgorithm
from ACO import AntColonyOptimization


def create_algorithm(algorithm_type: str, dist_mat, config):
    """创建算法实例

    Args:
        algorithm_type: 算法类型 ('ga' 或 'aco')
        dist_mat: 距离矩阵
        config: 配置参数

    Returns:
        TSPAlgorithm: 算法实例

    Raises:
        ValueError: 如果算法类型不支持
    """
    if algorithm_type == 'ga':
        return GeneticAlgorithm(dist_mat, config)
    elif algorithm_type == 'aco':
        return AntColonyOptimization(dist_mat, config)
    else:
        raise ValueError(f"不支持的算法类型: {algorithm_type}")