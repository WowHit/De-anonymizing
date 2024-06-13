import numpy as np
from scipy.optimize import linear_sum_assignment

# 计算最短路径距离矩阵
def compute_distance_matrix(network):
    size = len(network)  # 获取网络的大小（节点数）
    
    # 初始化距离矩阵，所有元素初始化为正无穷
    dist = np.full((size, size), np.inf)
    
    # 对角线元素初始化为0，表示节点到自身的距离为0
    np.fill_diagonal(dist, 0)

    # 遍历网络中的每条边
    for i in range(size):
        for j in range(size):
            if network[i][j] != 0:  # 如果存在边(i, j)
                dist[i][j] = network[i][j]  # 初始化该边的距离

    # 使用Floyd-Warshall算法计算最短路径距离矩阵
    for k in range(size):  # 遍历所有中间节点k
        for i in range(size):  # 遍历所有起点i
            for j in range(size):  # 遍历所有终点j
                # 如果通过节点k可以找到更短的路径，则更新dist[i][j]
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist  # 返回计算出的最短路径距离矩阵


# 计算两个网络节点之间的匹配关系
def CDA(DA, DB, l):
    cost_matrix = np.abs(DA - DB.T) # 计算代价矩阵
    row_ind, col_ind = linear_sum_assignment(cost_matrix) # 使用linear_sum_assignment(匈牙利算法)函数找到使总匹配代价最小的匹配组合
    
    matching = {}
    for i in range(len(row_ind)): # 遍历匹配结果，只有当匹配代价小于等于阈值 l 时，才将该匹配加入到结果中
        if cost_matrix[row_ind[i], col_ind[i]] <= l:
            matching[row_ind[i]] = col_ind[i]
    
    return matching


# 使用CDA计算两个网络的最优全匹配
def optimal_full_matching(A, B):
    DA = compute_distance_matrix(A)
    DB = compute_distance_matrix(B)
    l = max(np.max(DA), np.max(DB))
    return CDA(DA, DB, l)

# 计算给定匹配的精度和召回率
def link_precision_and_recall(matching, A, B):
    matched_edges_A = set()
    matched_edges_B = set()

    # 遍历匹配结果中的每个节点对 (u, v)
    for u in matching:
        for v in range(A.shape[1]):
            # 检查从网络 A 中的节点 u 到 v 的边是否存在，并且匹配中的 v 在网络 B 中的匹配节点是否在范围内
            if A[u, v] == 1 and v in matching and matching[v] in range(B.shape[1]) and B[matching[u], matching[v]] == 1:
                # 如果边存在，则添加到匹配边集合中
                matched_edges_A.add((u, v))
                matched_edges_B.add((matching[u], matching[v]))
                
    # 计算真阳性（True Positive）在两个匹配边集合中的交集大小
    true_positive = len(matched_edges_A & matched_edges_B)
    
    # 计算可能的阳性数
    possible_positive_A = len(matched_edges_A)
    possible_positive_B = len(matched_edges_B)

    # 计算精度 (Precision)
    precision = true_positive / possible_positive_A if possible_positive_A else 0
    
    # 计算召回率 (Recall)
    recall = true_positive / possible_positive_B if possible_positive_B else 0

    return precision, recall


# 计算f_score
def structural_f_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# 计算距离矩阵： 首先调用 compute_distance_matrix 函数计算网络 A 和 B 的距离矩阵 DA 和 DB。
# 初始化成本矩阵： 将 l 设置为 DA 和 DB 中的最大值，并调用 CDA 函数得到初始匹配 π。
# 计算初始 F 分数： 计算初始匹配 π 的结构 F 分数 f1。
# 迭代优化匹配： 对于每个节点对 (uA, uB)，尝试删除 π 中的一个节点，并计算更新后的匹配 π_temp 的结构 F 分数 f2。如果 f2 比当前最佳的结构 F 分数 f1 更高，则更新 π 和 f1。
# 返回最优匹配： 返回经过优化后的最优匹配 π。
def DDA(A, B):
    DA = compute_distance_matrix(A)
    DB = compute_distance_matrix(B)
    l = max(np.max(DA), np.max(DB))

    π = CDA(DA, DB, l)
    f1 = structural_f_score(*link_precision_and_recall(π, A, B))

    for uA, uB in list(π.items()):
        π_temp = π.copy()
        del π_temp[uA]
        f2 = structural_f_score(*link_precision_and_recall(π_temp, A, B))

        if f2 > f1:
            π = π_temp
            f1 = f2

    return π

# 初始化匹配： 从种子匹配 seed_matching 开始，复制到 matching 中，并初始化已匹配节点集合 matched_nodes。
# 迭代优化匹配： 使用 while 循环持续优化匹配，直到不能再更新为止。
# 节点匹配尝试： 对于每个节点 u，如果 u 还没有被匹配，则尝试将其匹配到网络 B 中的某个节点 v。
# 计算结构 F 分数： 对于每个匹配尝试，计算临时匹配 temp_matching 的精度和召回率，并计算 f_score。
# 接受匹配： 如果 f_score 大于当前最佳的结构 F 分数 best_f_score，则更新最佳匹配和结构 F 分数，并将匹配结果更新到 matching 和 matched_nodes 中。
# 检查渗透阈值： 每次接受匹配后，检查 best_f_score 是否达到或超过 percolation_threshold。如果是，则认为这个匹配是有效的，跳出内层循环，否则继续尝试其他匹配。
# 结束循环： 如果没有更新过匹配 (updated 为 False)，则退出循环，返回最终的匹配结果 matching。
def pda(seed_matching, A, B, percolation_threshold):
    matched_nodes = set(seed_matching.values())
    matching = seed_matching.copy()
    while True:
        updated = False
        for u in range(A.shape[0]):
            if u not in matching:
                best_match = None
                best_f_score = 0
                for v in range(B.shape[0]):
                    if v not in matching.values():
                        temp_matching = matching.copy()
                        temp_matching[u] = v
                        precision, recall = link_precision_and_recall(temp_matching, A, B)
                        f_score = structural_f_score(precision, recall)
                        if f_score > best_f_score:
                            best_f_score = f_score
                            best_match = v
                if best_match is not None and best_f_score >= percolation_threshold:
                    matching[u] = best_match
                    matched_nodes.add(best_match)
                    updated = True
                    break  # 跳出内层循环，认为这个匹配是有效的
        if not updated:
            break
    return matching

# 测试
if __name__ == "__main__":
    test_cases = [
        ("Test Case 1", np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0]
        ]), np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0]
        ])),
        ("Test Case 2", np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0]
        ]), np.array([
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 1],
            [1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ])),
        ("Test Case 3", np.array([
            [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
        ]), np.array([
            [0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
        ]))
    ]

    # 设置渗透阈值
    percolation_threshold = 0.5

    for case_name, A, B in test_cases:
        print(case_name)
        # 执行DDA算法
        dda_matching = DDA(A, B)
        print("DDA 匹配结果:", dda_matching)

        # 执行PDA算法
        pda_matching = pda(dda_matching, A, B, percolation_threshold)
        print("PDA 匹配结果:", pda_matching)
        print()
