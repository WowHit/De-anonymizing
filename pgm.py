import networkx as nx

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

def percolation_graph_matching(GA, GB, seed_set, threshold):
    matching = initialize_matching(seed_set)
    A_star = set(matching.keys())
    B_star = set(matching.values())
    
    while True:
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
    
    return matching

# 示例
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

# 种子节点
seed_set = [(1, 3)]

matching = percolation_graph_matching(G_A, G_B, seed_set, threshold=1)
print("Matching Result:", matching)


# 种子节点的选择（no）
"""import networkx as nx

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

def select_seed_nodes(GA, GB, seed_size):
    degree_sequence_A = sorted(GA.degree, key=lambda x: x[1], reverse=True)
    degree_sequence_B = sorted(GB.degree, key=lambda x: x[1], reverse=True)
    
    seed_set = []
    for i in range(seed_size):
        u = degree_sequence_A[i][0]
        v = degree_sequence_B[i][0]
        seed_set.append((u, v))
    
    print("Seed nodes:", seed_set)  # 打印选定的种子节点
    return seed_set

def percolation_graph_matching(GA, GB, seed_size, threshold):
    seed_set = select_seed_nodes(GA, GB, seed_size)
    matching = initialize_matching(seed_set)
    A_star = set(matching.keys())
    B_star = set(matching.values())
    
    while True:
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
    
    return matching

# Define the graphs G_A and G_B based on the user's example
G_A = nx.Graph({1: [2, 4], 2: [1], 3: [4], 4: [1, 3, 5], 5: [4]})
G_B = nx.Graph({1: [4], 2: [3], 3: [2, 4], 4: [1, 3, 5], 5: [4]})

# Set the seed size and percolation threshold
seed_size = 1
threshold = 1

# Perform the percolation graph matching
matching = percolation_graph_matching(G_A, G_B, seed_size, threshold)
print("Matching Result:", matching)
"""
