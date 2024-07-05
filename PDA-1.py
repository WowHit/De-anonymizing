import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

# 定义辅助函数

def compute_distance_matrix(nodes, links):
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

    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if distance_matrix[i, j] > distance_matrix[i, k] + distance_matrix[k, j]:
                    distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]

    return distance_matrix, node_index

def compute_multi_hop_adjacency_discrepancy(distance_matrix_A, distance_matrix_B, level):
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
    cost_matrix = np.zeros((DA.shape[0], DB.shape[0]))

    for i in range(DA.shape[0]):
        for j in range(DB.shape[0]):
            cost_matrix[i, j] = np.sum(np.abs(DA[i] - DB[j]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    π = np.zeros((DA.shape[0], DB.shape[0]))
    π[row_ind, col_ind] = 1

    return π

def cda(distance_matrix_A, distance_matrix_B, level):
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
    high_degree_nodes = [node for node, degree in G.degree() if degree >= degree_threshold]
    return high_degree_nodes

def f_score(matching, A, B):
    true_positive_edges = 0
    all_possible_edges_A = 0
    all_possible_edges_B = 0

    for u in A.nodes:
        for v in A.nodes:
            if u in matching and v in matching and A.has_edge(u, v) and B.has_edge(matching[u], matching[v]):
                true_positive_edges += 1
            if A.has_edge(u, v):
                all_possible_edges_A += 1
            if B.has_edge(u, v):
                all_possible_edges_B += 1

    link_precision = true_positive_edges / all_possible_edges_A if all_possible_edges_A else 0
    link_recall = true_positive_edges / all_possible_edges_B if all_possible_edges_B else 0
    f_score_value = (2 * link_precision * link_recall) / (link_precision + link_recall) if (link_precision + link_recall) else 0
    return link_precision, link_recall, f_score_value

def greedy_optimization(matching, f_score_func, A, B):
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
    high_degree_nodes_A = find_high_degree_nodes(GA, dT)
    high_degree_nodes_B = find_high_degree_nodes(GB, dT)

    sub_GA = GA.subgraph(high_degree_nodes_A)
    sub_GB = GB.subgraph(high_degree_nodes_B)

    distance_matrix_A, node_index_A = compute_distance_matrix(list(sub_GA.nodes), nx.to_dict_of_lists(sub_GA))
    distance_matrix_B, node_index_B = compute_distance_matrix(list(sub_GB.nodes), nx.to_dict_of_lists(sub_GB))

    π_high_degree = cda(distance_matrix_A, distance_matrix_B, level=np.max([distance_matrix_A.max(), distance_matrix_B.max()]))

    seed_set_high_degree = [(list(sub_GA.nodes)[idx_A], list(sub_GB.nodes)[idx_B]) for idx_A, idx_B in np.argwhere(π_high_degree == 1)]

    matching = {node_A: node_B for node_A, node_B in seed_set_high_degree}

    print(f"\nInitial high-degree matching matrix π_high_degree:\n{π_high_degree}")
    print(f"Seed set high-degree nodes: {seed_set_high_degree}")

    initial_precision, initial_recall, initial_f_score = f_score(matching, GA, GB)
    print(f"\nInitial F-score: {initial_f_score:.2f}, Precision: {initial_precision:.2f}, Recall: {initial_recall:.2f}")

    matching = percolation_graph_matching(GA, GB, seed_set_high_degree, r)

    final_precision, final_recall, final_f_score = f_score(matching, GA, GB)
    print(f"\nFinal F-score: {final_f_score:.2f}, Precision: {final_precision:.2f}, Recall: {final_recall:.2f}")

    return matching

if __name__ == "__main__":
    # 定义两个示例图
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

    # 设置度数阈值 dT 和迭代轮次数 r
    dT = 2
    r = 10

    # 调用 PDA 函数进行图匹配
    matching_with_high_degree_nodes = PDA(G_A, G_B, dT, r)

    # 输出匹配结果
    print("\nMatching results with high-degree nodes and PGM extension:")
    print(matching_with_high_degree_nodes)

    # 计算并输出 F-score、Precision 和 Recall
    precision, recall, f_score_value = f_score(matching_with_high_degree_nodes, G_A, G_B)
    print(f"\nF-score: {f_score_value:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
