import argparse


def create_parser():
    """创建并配置参数解析器

    Returns:
        ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser(description='遗传算法求解旅行商问题的配置参数')

    # 数据相关参数
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--city_num', type=int, default=20,
                            help='城市数量，默认20')
    data_group.add_argument('--pos_dimension', type=int, default=2,
                            help='坐标维度，默认2（二维坐标）')
    data_group.add_argument('--individual_num', type=int, default=50,
                            help='种群大小（个体数量），默认50')
    data_group.add_argument('--gen_num', type=int, default=200,
                            help='进化代数，默认200')
    data_group.add_argument('--mutate_prob', type=float, default=0.2,
                            help='变异概率，默认0.2')
    data_group.add_argument('--cross_prob', type=float, default=0.8,
                            help='交叉概率，默认0.8')
    data_group.add_argument('--tournament_size', type=int, default=5,
                            help='锦标赛选择中的小组大小，默认5')

    return parser


# 创建解析器实例
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
    print('* 城市数量:', config.city_num)
    print('* 坐标维度:', config.pos_dimension)
    print('* 种群大小:', config.individual_num)
    print('* 进化代数:', config.gen_num)
    print('* 变异概率:', config.mutate_prob)
    print('* 交叉概率:', config.cross_prob)
    print('* 锦标赛大小:', config.tournament_size)