import numpy
import config as conf
from init import create_algorithm
import matplotlib.pyplot as plt
import time


def build_distance_matrix(points: numpy.ndarray) -> numpy.ndarray:
    """构建城市间距离矩阵

    Args:
        points: 城市坐标矩阵，形状为(n, d)，n为城市数量，d为坐标维度

    Returns:
        np.ndarray: 距离矩阵，形状为(n, n)
    """
    # 使用向量化计算提高效率
    diff = points[:, numpy.newaxis, :] - points[numpy.newaxis, :, :]  # 计算坐标差
    dist_mat = numpy.sqrt(numpy.sum(diff ** 2, axis=-1))  # 计算欧氏距离
    return dist_mat


def main():
    """主函数"""
    # 获取配置参数
    config = conf.get_config()
    conf.print_config(config)

    # 记录开始时间
    start_time = time.time()

    # 随机生成城市坐标
    print(f"\n随机生成 {config.city_num} 个城市坐标...")
    city_positions = numpy.random.rand(config.city_num, config.pos_dimension) * 100

    # 定义城市名称（使用编号）
    city_names = [f"城市{i}" for i in range(config.city_num)]

    # 计算城市距离矩阵
    print("计算距离矩阵...")
    dist_matrix = build_distance_matrix(city_positions)

    # 创建算法实例
    print(f"初始化{'遗传算法' if config.algorithm == 'ga' else '蚁群算法'}...")
    algorithm = create_algorithm(config.algorithm, dist_matrix, config)

    # 运行算法
    print("开始计算过程...")
    result_list, fitness_list = algorithm.run()

    # 获取最终结果
    best_path, best_distance = algorithm.get_result()
    final_path = city_positions[best_path, :]

    # 记录结束时间
    end_time = time.time()
    print(f"\n算法运行时间: {end_time - start_time:.2f} 秒")
    print(f"最短路径长度: {best_distance:.4f}")

    # 可视化结果
    plt.rcParams['font.sans-serif'] = ['Hei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 绘制最优路径
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(final_path[:, 0], final_path[:, 1], 'o-', linewidth=2, markersize=8)
    plt.plot([final_path[-1, 0], final_path[0, 0]],
             [final_path[-1, 1], final_path[0, 1]], 'o-', linewidth=2, markersize=8)

    # 添加城市名称标签
    for i, (x, y) in enumerate(final_path):
        # 获取城市名称
        city_name = city_names[best_path[i]]
        # 添加标签，稍微偏移以避免与点重叠
        plt.annotate(city_name, (x, y), xytext=(5, 5), textcoords='offset points',
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))

    algorithm_name = "遗传算法" if config.algorithm == "ga" else "蚁群算法"
    plt.title(f'{algorithm_name} - 最优路径 (城市数量: {config.city_num})')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.grid(True)

    # 绘制适应度曲线
    plt.subplot(1, 2, 2)
    plt.plot(fitness_list, linewidth=2)
    plt.title(f'{algorithm_name} - 适应度变化曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('路径长度')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'TSP_result_{config.algorithm}_{config.city_num}cities.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印最优路径的城市顺序
    print(f"\n{algorithm_name}最优路径的城市顺序:")
    for i, city_index in enumerate(best_path):
        print(f"{i + 1}. {city_names[city_index]}")

    # 打印路径总长度
    print(f"\n路径总长度: {best_distance:.2f}")


if __name__ == "__main__":
    main()