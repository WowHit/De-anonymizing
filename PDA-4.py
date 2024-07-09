import networkx as nx
import numpy as np
import random
from scipy.optimize import linear_sum_assignment

def compute_distance_matrix(nodes, links):
    """
    计算图的最短路径距离矩阵。
    
    参数:
        nodes (list): 图中的节点列表。
        links (dict): 图的邻接表表示。
    
    返回:
        distance_matrix (np.ndarray): 最短路径距离矩阵。
        node_index (dict): 节点与其在矩阵中索引的映射。
    """
    all_nodes = sorted(set(nodes).union(*links.values()))
    num_nodes = len(all_nodes)
    node_index = {node: idx for idx, node in enumerate(all_nodes)}
    distance_matrix = np.full((num_nodes, num_nodes), np.inf)

    for node in nodes:
        distance_matrix[node_index[node], node_index[node]] = 0
        if node in links:
            for neighbor in links[node]:
                if neighbor in node_index:
                    distance_matrix[node_index[node], node_index[neighbor]] = 1

    # 使用Floyd-Warshall算法计算所有节点对之间的最短路径
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if distance_matrix[i, j] > distance_matrix[i, k] + distance_matrix[k, j]:
                    distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]

    return distance_matrix, node_index

def compute_multi_hop_adjacency_discrepancy(distance_matrix_A, distance_matrix_B, level):
    """
    计算多跳邻接矩阵差异。
    
    参数:
        distance_matrix_A (np.ndarray): 图A的距离矩阵。
        distance_matrix_B (np.ndarray): 图B的距离矩阵。
        level (int): 多跳级别。
    
    返回:
        DA, DB (np.ndarray): 图A和图B的多跳邻接矩阵。
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
    使用线性和指派算法优化匹配。
    
    参数:
        DA (np.ndarray): 图A的多跳邻接矩阵。
        DB (np.ndarray): 图B的多跳邻接矩阵。
    
    返回:
        π (np.ndarray): 优化后的匹配矩阵。
    """
    num_nodes_A, num_nodes_B = DA.shape[0], DB.shape[0]
    
    # 确保两个图的节点数一致，通过填充使两个矩阵的大小一致
    if num_nodes_A != num_nodes_B:
        max_size = max(num_nodes_A, num_nodes_B)
        DA = np.pad(DA, ((0, max_size - num_nodes_A), (0, max_size - num_nodes_A)), mode='constant', constant_values=0)
        DB = np.pad(DB, ((0, max_size - num_nodes_B), (0, max_size - num_nodes_B)), mode='constant', constant_values=0)

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
    迭代优化匹配矩阵。
    
    参数:
        distance_matrix_A (np.ndarray): 图A的距离矩阵。
        distance_matrix_B (np.ndarray): 图B的距离矩矩。
        level (int): 多跳级别。
    
    返回:
        π (np.ndarray): 优化后的匹配矩阵。
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
    找出高度节点。
    
    参数:
        G (networkx.Graph): 输入图。
        degree_threshold (int): 度阈值。
    
    返回:
        high_degree_nodes (list): 高度节点列表。
    """
    nodes_by_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)
    high_degree_nodes = [node for node, degree in nodes_by_degree if degree >= degree_threshold]
    return high_degree_nodes

def f_score(matching, A, B):
    """
    计算F1评分、精确率和召回率。
    
    参数:
        matching (dict): 匹配结果。
        A, B (networkx.Graph): 两个输入图。
    
    返回:
        link_precision, link_recall, f_score_value (float): 精确率、召回率和F1评分。
    """
    true_positive_edges = 0
    all_possible_edges_A = sum(1 for u in matching for v in matching if A.has_edge(u, v))
    all_possible_edges_B = sum(1 for u in matching for v in matching if B.has_edge(matching[u], matching[v]))

    for u in matching:
        for v in matching:
            if A.has_edge(u, v) and B.has_edge(matching[u], matching[v]):
                true_positive_edges += 1

    link_precision = true_positive_edges / all_possible_edges_A if all_possible_edges_A else 0
    link_recall = true_positive_edges / all_possible_edges_B if all_possible_edges_B else 0
    f_score_value = (2 * link_precision * link_recall) / (link_precision + link_recall) if (
            link_precision + link_recall) else 0
    return link_precision, link_recall, f_score_value

def greedy_optimization(matching, f_score_func, A, B):
    """
    通过贪婪优化改进匹配结果。
    
    参数:
        matching (dict): 初始匹配结果。
        f_score_func (function): 计算F1评分的函数。
        A, B (networkx.Graph): 两个输入图。
    
    返回:
        best_matching (dict): 优化后的匹配结果。
    """
    best_f_score = 0
    best_matching = matching.copy()

    for uA in list(matching.keys()):
        for uB in list(B.nodes):
            if uB in matching.values():
                continue
            new_matching = matching.copy()
            new_matching[uA] = uB
            _, _, new_f_score = f_score_func(new_matching, A, B)
            if new_f_score > best_f_score:
                best_f_score = new_f_score
                best_matching = new_matching

    return best_matching

def initialize_matching(GA, GB, num_seeds):
    """
    初始化种子节点匹配。
    
    参数:
        GA, GB (networkx.Graph): 两个输入图。
        num_seeds (int): 种子节点数量。
    
    返回:
        matching (dict): 初始匹配结果。
    """
    common_nodes = list(set(GA.nodes) & set(GB.nodes))
    seed_set = random.sample(common_nodes, num_seeds)
    matching = {}

    for node in seed_set:
        neighbors_A = list(GA.neighbors(node))
        neighbors_B = list(GB.neighbors(node))

        if neighbors_A and neighbors_B:
            match_A = random.choice(neighbors_A)
            match_B = random.choice(neighbors_B)

            matching[node] = match_A
            matching[match_A] = node
            matching[match_B] = node
            matching[node] = match_B

    return matching

def find_candidate_matches(GA, GB, A_star, B_star, threshold):
    """
    找出候选匹配对。
    
    参数:
        GA, GB (networkx.Graph): 两个输入图。
        A_star, B_star (set): 当前已匹配节点集。
        threshold (int): 邻居阈值。
    
    返回:
        candidate_pairs (list): 候选匹配对列表。
    """
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

def pgm_algorithm(GA, GB, num_seeds, threshold):
    """
    实现PGM算法。
    
    参数:
        GA, GB (networkx.Graph): 两个输入图。
        num_seeds (int): 种子节点数量。
        threshold (int): 邻居阈值。
    
    返回:
        matching (dict): 最终匹配结果。
    """
    matching = initialize_matching(GA, GB, num_seeds)
    A_star = set(matching.keys())
    B_star = set(matching.values())

    while True:
        candidate_pairs = find_candidate_matches(GA, GB, A_star, B_star, threshold)
        if not candidate_pairs:
            break

        new_matching = {}

        for u, v in candidate_pairs:
            neighbors_A = set(GA.neighbors(u))
            neighbors_B = set(GB.neighbors(v))

            common_neighbors = neighbors_A & A_star
            common_neighbors_mapped = {matching[neighbor] for neighbor in common_neighbors if neighbor in matching}

            if len(common_neighbors_mapped) >= threshold:
                new_matching[u] = v
                A_star.add(u)
                B_star.add(v)

        if not new_matching:
            break

        matching.update(new_matching)

    return matching

# 示例用法
GA = nx.erdos_renyi_graph(100, 0.1)
GB = nx.erdos_renyi_graph(100, 0.1)

matching = pgm_algorithm(GA, GB, num_seeds=10, threshold=3)
print("PGM匹配结果：", matching)

distance_matrix_A, node_index_A = compute_distance_matrix(list(GA.nodes), dict(GA.adjacency()))
distance_matrix_B, node_index_B = compute_distance_matrix(list(GB.nodes), dict(GB.adjacency()))

cda_result = cda(distance_matrix_A, distance_matrix_B, level=2)
print("CDA匹配结果：", cda_result)

matching_fscore = initialize_matching(GA, GB, num_seeds=10)
optimized_matching = greedy_optimization(matching_fscore, f_score, GA, GB)
print("优化后的F-score匹配结果：", optimized_matching)
