import argparse


def create_parser():
    """创建并配置参数解析器

    Returns:
        ArgumentParser: 配置好的参数解析器
    """
    # 使用不同的变量名，避免与全局变量冲突
    local_parser = argparse.ArgumentParser(description='多算法求解旅行商问题的配置参数')

    # 算法选择参数
    algo_group = local_parser.add_argument_group('Algorithm')
    algo_group.add_argument('--algorithm', type=str, default='ga', choices=['ga', 'aco'],
                            help='选择算法: ga(遗传算法)或aco(蚁群算法)，默认ga')

    # 数据相关参数
    data_group = local_parser.add_argument_group('Data')
    data_group.add_argument('--city_num', type=int, default=20,
                            help='城市数量，默认20')
    data_group.add_argument('--pos_dimension', type=int, default=2,
                            help='坐标维度，默认2（二维坐标）')

    # 遗传算法参数
    ga_group = local_parser.add_argument_group('Genetic Algorithm')
    ga_group.add_argument('--ga_individual_num', type=int, default=50,
                          help='遗传算法种群大小（个体数量），默认50')
    ga_group.add_argument('--ga_gen_num', type=int, default=200,
                          help='遗传算法进化代数，默认200')
    ga_group.add_argument('--ga_mutate_prob', type=float, default=0.2,
                          help='遗传算法变异概率，默认0.2')
    ga_group.add_argument('--ga_cross_prob', type=float, default=0.8,
                          help='遗传算法交叉概率，默认0.8')
    ga_group.add_argument('--ga_tournament_size', type=int, default=5,
                          help='遗传算法锦标赛选择中的小组大小，默认5')

    # 蚁群算法参数
    aco_group = local_parser.add_argument_group('Ant Colony Optimization')
    aco_group.add_argument('--aco_ant_num', type=int, default=50,
                           help='蚁群算法蚂蚁数量，默认50')
    aco_group.add_argument('--aco_gen_num', type=int, default=200,
                           help='蚁群算法迭代次数，默认200')
    aco_group.add_argument('--aco_alpha', type=float, default=1.0,
                           help='蚁群算法信息素重要程度因子，默认1.0')
    aco_group.add_argument('--aco_beta', type=float, default=2.0,
                           help='蚁群算法启发函数重要程度因子，默认2.0')
    aco_group.add_argument('--aco_rho', type=float, default=0.5,
                           help='蚁群算法信息素挥发因子，默认0.5')
    aco_group.add_argument('--aco_q', type=float, default=100,
                           help='蚁群算法信息素增加强度系数，默认100')
    aco_group.add_argument('--aco_strategy', type=int, default=0,
                           help='蚁群算法策略: 0(蚂蚁周期系统), 1(蚂蚁数量系统), 2(蚂蚁密度系统)，默认0')

    return local_parser  # 返回局部变量


# 创建全局解析器实例
parser = create_parser()


def get_config():
    """获取解析后的配置参数

    Returns:
        Namespace: 包含所有配置参数的对象
    """
    config, unparsed = parser.parse_known_args()
    return config


def print_config(config):
    """打印配置参数

    Args:
        config: 配置参数对象
    """
    print('\n配置参数:')
    print(f'* 算法: {"遗传算法" if config.algorithm == "ga" else "蚁群算法"}')
    print('* 城市数量:', config.city_num)
    print('* 坐标维度:', config.pos_dimension)

    if config.algorithm == 'ga':
        print('\n遗传算法参数:')
        print('* 种群大小:', config.ga_individual_num)
        print('* 进化代数:', config.ga_gen_num)
        print('* 变异概率:', config.ga_mutate_prob)
        print('* 交叉概率:', config.ga_cross_prob)
        print('* 锦标赛大小:', config.ga_tournament_size)
    else:
        print('\n蚁群算法参数:')
        print('* 蚂蚁数量:', config.aco_ant_num)
        print('* 迭代次数:', config.aco_gen_num)
        print('* 信息素因子(α):', config.aco_alpha)
        print('* 启发因子(β):', config.aco_beta)
        print('* 挥发因子(ρ):', config.aco_rho)
        print('* 信息素强度(Q):', config.aco_q)
        print('* 策略:', ["蚂蚁周期系统", "蚂蚁数量系统", "蚂蚁密度系统"][config.aco_strategy])