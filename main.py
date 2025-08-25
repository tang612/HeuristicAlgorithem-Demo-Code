import numpy as np
import config as conf
from GA import GeneticAlgorithm
import matplotlib.pyplot as plt
import time


def build_distance_matrix(points: np.ndarray) -> np.ndarray:
    """构建城市间距离矩阵

    Args:
        points: 城市坐标矩阵，形状为(n, d)，n为城市数量，d为坐标维度

    Returns:
        np.ndarray: 距离矩阵，形状为(n, n)
    """
    n = points.shape[0]
    # 使用向量化计算提高效率
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # 计算坐标差
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1))  # 计算欧氏距离
    return dist_mat


def main():
    """主函数"""
    # 获取配置参数
    config = conf.get_config()
    conf.print_config(config)

    # 记录开始时间
    start_time = time.time()

    # 生成随机城市坐标
    print("\n生成城市坐标...")
    city_positions = np.random.rand(config.city_num, config.pos_dimension)

    """city_positions = np.array([
        [105, 38],  # 城市0
        [54, 14],  # 城市1
        [76, 65],  # 城市2
        [29, 96],  # 城市3
        [32, 43],  # 城市4
        [83, 78],  # 城市5
        [39, 24],  # 城市6
        [65, 50],  # 城市7
        [96, 75],  # 城市8
        [14, 36]  # 城市9
    ])"""

    # 计算城市距离矩阵
    print("计算距离矩阵...")
    dist_matrix = build_distance_matrix(city_positions)

    # 创建遗传算法实例
    print("初始化遗传算法...")
    ga = GeneticAlgorithm(dist_matrix, config)

    # 运行遗传算法
    print("开始进化过程...")
    result_list, fitness_list = ga.run()

    # 获取最终结果
    final_result = result_list[-1]
    final_path = city_positions[final_result, :]

    # 记录结束时间
    end_time = time.time()
    print(f"\n算法运行时间: {end_time - start_time:.2f} 秒")
    print(f"最短路径长度: {fitness_list[-1]:.4f}")

    # 可视化结果
    plt.rcParams['font.sans-serif'] = ['Hei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 绘制最优路径
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(final_path[:, 0], final_path[:, 1], 'o-', linewidth=2, markersize=8)
    plt.plot([final_path[-1, 0], final_path[0, 0]],
             [final_path[-1, 1], final_path[0, 1]], 'o-', linewidth=2, markersize=8)
    plt.title('最优路径')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.grid(True)

    # 绘制适应度曲线
    plt.subplot(1, 2, 2)
    plt.plot(fitness_list, linewidth=2)
    plt.title('适应度变化曲线')
    plt.xlabel('进化代数')
    plt.ylabel('路径长度')
    plt.grid(True)

    plt.tight_layout()
    #plt.savefig('TSP_result.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()