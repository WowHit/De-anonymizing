import numpy as np
from scipy.optimize import linear_sum_assignment

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


def find_high_degree_nodes(node_set, link_set, degree_threshold):
    high_degree_nodes = []
    for node in node_set:
        if len(link_set[node]) >= degree_threshold:
            high_degree_nodes.append(node)
    return high_degree_nodes

def compute_multi_hop_adjacency_discrepancy(GA, GB, level):
    DA = np.zeros((GA.shape[0], GA.shape[0]))
    DB = np.zeros((GB.shape[0], GB.shape[0]))
    
    for i in range(GA.shape[0]):
        for j in range(GA.shape[1]):
            DA[i, j] = np.sum(GA[i, :] <= level) - np.sum(GA[j, :] <= level)
    
    for i in range(GB.shape[0]):
        for j in range(GB.shape[1]):
            DB[i, j] = np.sum(GB[i, :] <= level) - np.sum(GB[j, :] <= level)
    
    return DA, DB

def optimize_matching(DA, DB):
    large_value = 1e6

    min_dim = min(DA.shape[0], DB.shape[0])
    cost_matrix = np.zeros((DA.shape[0], DB.shape[0]))

    for i in range(DA.shape[0]):
        for j in range(DB.shape[0]):
            cost_matrix[i, j] = np.abs(DA[i] - DB[j]).sum()

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    π = np.zeros((DA.shape[0], DB.shape[0]))
    π[row_ind, col_ind] = 1
    
    return π

def cda(distance_matrix_A, distance_matrix_B, level):
    DA, DB = compute_multi_hop_adjacency_discrepancy(distance_matrix_A, distance_matrix_B, level)
    π = np.zeros((distance_matrix_A.shape[0], distance_matrix_B.shape[0]))

    while True:
        print("\nCurrent Matching Matrix:")
        print(π)

        π_new = optimize_matching(DA, DB)

        if np.array_equal(π, π_new):
            break
        else:
            π = π_new

    return π

# 示例
G_A = {
    1: [2, 4],
    2: [1],
    3: [4],
    4: [1, 3, 5],
    5: [4]
}

G_B = {
    1: [4],
    2: [3],
    3: [2, 4],
    4: [1, 3, 5],
    5: [4]
}

nodes_A = list(G_A.keys())
nodes_B = list(G_B.keys())

distance_matrix_A, node_index_A = compute_distance_matrix(nodes_A, G_A)
distance_matrix_B, node_index_B = compute_distance_matrix(nodes_B, G_B)

print("Distance Matrix A:")
print(distance_matrix_A)
print("\nDistance Matrix B:")
print(distance_matrix_B)

level = 2
matching = cda(distance_matrix_A, distance_matrix_B, level)
print("\nOptimal Matching Matrix:")
print(matching)
