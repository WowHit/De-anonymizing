import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
import random

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
        distance_matrix_B (np.ndarray): 图B的距离矩阵。
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

def calculate_score(u, v, GA, GB, matching):
    """
    计算候选匹配对的得分。
    
    参数:
        u, v (int): 候选匹配对的节点。
        GA, GB (networkx.Graph): 两个输入图。
        matching (dict): 当前匹配结果。
    
    返回:
        score (int): 匹配得分。
    """
    neighbors_u = set(GA.neighbors(u))
    neighbors_v = set(GB.neighbors(v))
    matched_neighbors = set(matching[n] for n in neighbors_u if n in matching)
    return len(matched_neighbors & neighbors_v)

def select_best_candidate(candidates, GA, GB, matching):
    """
    从候选匹配对中选择最佳匹配。
    
    参数:
        candidates (list): 候选匹配对列表。
        GA, GB (networkx.Graph): 两个输入图。
        matching (dict): 当前匹配结果。
    
    返回:
        best_match (tuple): 最佳匹配对。
    """
    best_match = None
    best_score = -1
    for u, v in candidates:
        score = calculate_score(u, v, GA, GB, matching)
        if score > best_score:
            best_score = score
            best_match = (u, v)
    return best_match

def percolation_graph_matching(GA, GB, seed_set, r, threshold):
    """
    执行渗透图匹配算法。
    
    参数:
        GA, GB (networkx.Graph): 两个输入图。
        seed_set (list): 种子节点集。
        r (int): 渗透半径。
        threshold (int): 邻居阈值。
    
    返回:
        matching (dict): 最终匹配结果。
    """
    matching = initialize_matching(GA, GB, len(seed_set))
    A_star = set(matching.keys())
    B_star = set(matching.values())

    while r > 0:
        candidate_matches = find_candidate_matches(GA, GB, A_star, B_star, threshold)
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
    """
    执行种子驱动对齐算法 (PDA)。
    
    参数:
        GA, GB (networkx.Graph): 两个输入图。
        dT (int): 度阈值。
        r (int): 渗透半径。
    
    返回:
        matching (dict): 最终匹配结果。
    """
    high_degree_nodes_A = find_high_degree_nodes(GA, dT)
    high_degree_nodes_B = find_high_degree_nodes(GB, dT)

    common_high_degree_nodes = set(high_degree_nodes_A) & set(high_degree_nodes_B)
    if not common_high_degree_nodes:
        raise ValueError("No common high-degree nodes found between GA and GB.")

    sub_GA = GA.subgraph(common_high_degree_nodes)
    sub_GB = GB.subgraph(common_high_degree_nodes)

    distance_matrix_A, node_index_A = compute_distance_matrix(list(sub_GA.nodes), nx.to_dict_of_lists(sub_GA))
    distance_matrix_B, node_index_B = compute_distance_matrix(list(sub_GB.nodes), nx.to_dict_of_lists(sub_GB))

    π_high_degree = cda(distance_matrix_A, distance_matrix_B,
                        level=np.max([distance_matrix_A.max(), distance_matrix_B.max()]))

    seed_set_high_degree = [(list(sub_GA.nodes)[idx_A], list(sub_GB.nodes)[idx_B]) for idx_A, idx_B in
                            np.argwhere(π_high_degree == 1)]

    matching_high_degree = {node_A: node_B for node_A, node_B in seed_set_high_degree}

    matching = percolation_graph_matching(GA, GB, matching_high_degree, r, threshold=1)

    return matching

def generate_random_graph(num_nodes, num_edges, high_degree_nodes):
    """
    生成随机图。
    
    参数:
        num_nodes (int): 节点数。
        num_edges (int): 边数。
        high_degree_nodes (list): 高度节点列表。
    
    返回:
        G (networkx.Graph): 生成的随机图。
    """
    G = nx.gnm_random_graph(num_nodes, num_edges)

    for node in high_degree_nodes:
        for i in range(num_nodes):
            if i != node:
                G.add_edge(node, i)

    return G

def generate_test_data(num_samples, num_nodes, num_edges, num_high_degree_nodes):
    """
    生成测试数据。
    
    参数:
        num_samples (int): 样本数。
        num_nodes (int): 每个图的节点数。
        num_edges (int): 每个图的边数。
        num_high_degree_nodes (int): 每个图的高度节点数。
    
    返回:
        test_data (list): 测试数据列表，每个元素为两个图的元组。
    """
    test_data = []
    for _ in range(num_samples):
        high_degree_nodes = random.sample(range(num_nodes), num_high_degree_nodes)
        G_A = generate_random_graph(num_nodes, num_edges, high_degree_nodes)
        G_B = generate_random_graph(num_nodes, num_edges, high_degree_nodes)

        test_data.append((G_A, G_B))
    return test_data

def print_adj_matrices(test_data):
    """
    打印测试数据的邻接矩阵。
    
    参数:
        test_data (list): 测试数据列表，每个元素为两个图的元组。
    """
    for idx, (G_A, G_B) in enumerate(test_data):
        print(f"\nTest Sample {idx + 1}:")
        print("Graph A adjacency matrix:")
        print(nx.adjacency_matrix(G_A).todense())
        print("\nGraph B adjacency matrix:")
        print(nx.adjacency_matrix(G_B).todense())

if __name__ == "__main__":
    num_samples = 5
    num_nodes = 20
    num_edges = 8
    num_high_degree_nodes = 2

    test_data = generate_test_data(num_samples, num_nodes, num_edges, num_high_degree_nodes)
    print_adj_matrices(test_data)

    dT = 3
    r = 10

    for idx, (G_A, G_B) in enumerate(test_data):
        print(f"\nRunning PDA Algorithm on Test Sample {idx + 1}:")
        try:
            matching = PDA(G_A, G_B, dT, r)
            print(f"Matching result: {matching}")
            precision, recall, f_score_value = f_score(matching, G_A, G_B)
            print(f"F-score: {f_score_value:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        except ValueError as e:
            print(f"Error: {e}")
