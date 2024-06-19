import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_distance_matrix(nodes, links):
    """
    计算节点间的最短路径距离矩阵。

    参数:
    - nodes: 节点列表。
    - links: 节点邻居的字典。

    返回:
    - distance_matrix: 节点间最短路径距离矩阵。
    - node_index: 节点到索引的映射。
    """
    all_nodes = sorted(set(nodes).union(*links.values()))
    num_nodes = len(all_nodes)
    node_index = {node: idx for idx, node in enumerate(all_nodes)}
    distance_matrix = np.full((num_nodes, num_nodes), np.inf)

    # 填充对角线和直接连接
    for node in nodes:
        distance_matrix[node_index[node], node_index[node]] = 0
        if node in links:
            for neighbor in links[node]:
                if neighbor in node_index:
                    distance_matrix[node_index[node], node_index[neighbor]] = 1

    # 使用 Floyd-Warshall 算法计算最短路径
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if distance_matrix[i, j] > distance_matrix[i, k] + distance_matrix[k, j]:
                    distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]

    return distance_matrix, node_index

def compute_multi_hop_adjacency_discrepancy(distance_matrix_A, distance_matrix_B, level):
    """
    计算多跳邻接矩阵。

    参数:
    - distance_matrix_A: 图A的距离矩阵。
    - distance_matrix_B: 图B的距离矩阵。
    - level: 多跳邻接阈值。

    返回:
    - DA: 图A的多跳邻接矩阵。
    - DB: 图B的多跳邻接矩阵。
    """
    def multi_hop_adjacency(G, level):
        num_nodes = G.shape[0]
        hop_adj = np.zeros_like(G)
        for i in range(num_nodes):
            hop_adj[i, :] = (G[i, :] <= level).astype(int)
        return hop_adj

    DA = multi_hop_adjacency(distance_matrix_A, level)
    DB = multi_hop_adjacency(distance_matrix_B, level)
    return DA, DB

def optimize_matching(DA, DB):
    """
    使用线性分配问题优化匹配。

    参数:
    - DA: 图A的多跳邻接矩阵。
    - DB: 图B的多跳邻接矩阵。

    返回:
    - π: 最优匹配矩阵。
    """
    cost_matrix = np.zeros((DA.shape[0], DB.shape[0]))

    for i in range(DA.shape[0]):
        for j in range(DB.shape[0]):
            cost_matrix[i, j] = np.sum(np.abs(DA[i] - DB[j]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    π = np.zeros((DA.shape[0], DB.shape[0]))
    π[row_ind, col_ind] = 1

    return π

def cda(distance_matrix_A, distance_matrix_B, level):
    """
    执行CDA（基于跨图度量的算法）进行图匹配。

    参数:
    - distance_matrix_A: 图A的距离矩阵。
    - distance_matrix_B: 图B的距离矩阵。
    - level: 多跳邻接阈值。

    返回:
    - π: 最终匹配矩阵。
    """
    DA, DB = compute_multi_hop_adjacency_discrepancy(distance_matrix_A, distance_matrix_B, level)
    π = np.zeros((distance_matrix_A.shape[0], distance_matrix_B.shape[0]))

    while True:
        π_new = optimize_matching(DA, DB)

        if np.array_equal(π, π_new):
            break
        else:
            π = π_new

    return π

def find_high_degree_nodes(G, degree_threshold):
    """
    查找度大于等于阈值的节点。

    参数:
    - G: 图。
    - degree_threshold: 度阈值。

    返回:
    - high_degree_nodes: 度大于等于阈值的节点列表。
    """
    high_degree_nodes = [node for node, degree in G.degree() if degree >= degree_threshold]
    return high_degree_nodes

def f_score(matching, A, B):
    """
    基于匹配计算F-score。

    参数:
    - matching (dict): 匹配字典，键为图A的节点，值为图B的节点。
    - A (networkx.Graph): 图A。
    - B (networkx.Graph): 图B。

    返回:
    - link_precision (float): 精确度。
    - link_recall (float): 召回率。
    - f_score (float): F-score。
    """
    true_positive_edges = {(u, v) for u, v in matching.items() if A.has_edge(u, v) and B.has_edge(matching[u], v)}
    possible_edges_A = {(u, v) for u, v in matching.items()}
    link_precision = len(true_positive_edges) / len(possible_edges_A) if possible_edges_A else 0
    all_possible_edges_in_A = sum(1 for u in A.nodes for v in A.nodes if A.has_edge(u, v))
    link_recall = len(true_positive_edges) / all_possible_edges_in_A if all_possible_edges_in_A else 0
    f_score = (2 * link_precision * link_recall) / (link_precision + link_recall) if (link_precision + link_recall) else 0
    return link_precision, link_recall, f_score

def greedy_optimization(π, f_score_func, A, B):
    """
    基于F-score进行贪婪优化。

    参数:
    - π: 初始匹配矩阵。
    - f_score_func: 计算F-score的函数。
    - A: 图A。
    - B: 图B。

    返回:
    - optimal_π: 优化后的匹配矩阵。
    """
    f1_precision, f1_recall, f1_score = f_score_func(π, A, B)

    for uA, uB in np.argwhere(π == 1):
        π_temp = π.copy()
        π_temp[uA, uB] = 0
        f2_precision, f2_recall, f2_score = f_score_func(π_temp, A, B)

        if f2_score > f1_score:
            π = π_temp
            f1_precision, f1_recall, f1_score = f2_precision, f2_recall, f2_score

    return π

def initialize_matching(seed_set):
    matching = {}
    for (a, b) in seed_set:
        matching[a] = b
    return matching

def find_candidate_matches(GA, GB, A_star, B_star, threshold):
    candidate_pairs = []
    for u in GA.nodes:
        if u in A_star:
            continue
        for v in GB.nodes:
            if v in B_star:
                continue
            if len(set(GA.neighbors(u))) >= threshold and len(set(GB.neighbors(v))) >= threshold:
                candidate_pairs.append((u, v))
    return candidate_pairs

def calculate_score(u, v, GA, GB, matching):
    neighbors_u = set(GA.neighbors(u))
    neighbors_v = set(GB.neighbors(v))
    matched_neighbors = set(matching[n] for n in neighbors_u if n in matching)
    return len(matched_neighbors & neighbors_v)

def select_best_candidate(candidates, GA, GB, matching):
    best_match = None
    best_score = -1
    for u, v in candidates:
        score = calculate_score(u, v, GA, GB, matching)
        if score > best_score:
            best_score = score
            best_match = (u, v)
    return best_match

def percolation_graph_matching(GA, GB, seed_set, r):
    matching = initialize_matching(seed_set)
    A_star = set(matching.keys())
    B_star = set(matching.values())

    while r > 0:
        candidate_matches = find_candidate_matches(GA, GB, A_star, B_star, threshold=1)
        if not candidate_matches:
            break

        best_match = select_best_candidate(candidate_matches, GA, GB, matching)
        if not best_match:
            break

        u, v = best_match
        matching[u] = v
        A_star.add(u)
        B_star.add(v)
        r -= 1

    return matching

def PDA(GA, GB, dT, r):
    # Step 1: 找到两个图中度大于阈值的节点
    high_degree_nodes_A = find_high_degree_nodes(GA, dT)
    high_degree_nodes_B = find_high_degree_nodes(GB, dT)

    # Step 2: 创建高度节点的子图
    sub_GA = GA.subgraph(high_degree_nodes_A)
    sub_GB = GB.subgraph(high_degree_nodes_B)

    # Step 3: 计算子图的距离矩阵
    distance_matrix_A, node_index_A = compute_distance_matrix(list(sub_GA.nodes), nx.to_dict_of_lists(sub_GA))
    distance_matrix_B, node_index_B = compute_distance_matrix(list(sub_GB.nodes), nx.to_dict_of_lists(sub_GB))

    # Step 4: 在高度节点子图上执行CDA算法
    π_high_degree = cda(distance_matrix_A, distance_matrix_B, level=np.max([distance_matrix_A.max(), distance_matrix_B.max()]))

    # Step 5: 使用高度节点作为种子集进行匹配
    seed_set_high_degree = [(node_A, node_B) for idx_A, idx_B in np.argwhere(π_high_degree == 1)
                            for node_A in [list(sub_GA.nodes)[idx_A]]
                            for node_B in [list(sub_GB.nodes)[idx_B]]]

    # Step 6: 使用高度节点初始化匹配
    matching = {node_A: node_B for node_A, node_B in seed_set_high_degree}

    # Step 7: 执行PGM算法
    matching = percolation_graph_matching(GA, GB, seed_set_high_degree, r)

    return matching

if __name__ == "__main__":
    # 示例图
    G_A = nx.Graph({
        1: [2, 4],
        2: [1],
        3: [4],
        4: [1, 3, 5],
        5: [4]
    })

    G_B = nx.Graph({
        1: [4],
        2: [3],
        3: [2, 4],
        4: [1, 3, 5],
        5: [4]
    })

    # 参数设置
    dT = 2  # 高度节点的度阈值
    r = 10  # PGM的渗透阈值

    # 执行PDA获取匹配结果
    matching_with_high_degree_nodes = PDA(G_A, G_B, dT, r)

    # 打印匹配结果
    print("\n使用高度节点和PGM扩展的匹配结果:")
    print(matching_with_high_degree_nodes)

    # 评估F-score
    precision, recall, f_score = f_score(matching_with_high_degree_nodes, G_A, G_B)
    print(f"\nF-score: {f_score:.2f}, 精确度: {precision:.2f}, 召回率: {recall:.2f}")
