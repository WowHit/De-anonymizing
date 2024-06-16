def compute_link_overlap(E_pi_A, E_pi_B):
    intersection = set(E_pi_A) & set(E_pi_B)
    return len(intersection)

def compute_link_union(E_pi_A, E_pi_B):
    union = set(E_pi_A) | set(E_pi_B)
    return len(union)

def link_precision(E_pi_A, E_pi_B):
    overlap = compute_link_overlap(E_pi_A, E_pi_B)
    union = compute_link_union(E_pi_A, E_pi_B)
    if union == 0:
        return 0
    return overlap / union

def link_recall(E_pi_A, E_pi_B, E_pi_f_A, E_pi_f_B):
    overlap = compute_link_overlap(E_pi_A, E_pi_B)
    full_overlap = compute_link_overlap(E_pi_f_A, E_pi_f_B)
    if full_overlap == 0:
        return 0
    return overlap / full_overlap

def link_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# 示例数据
E_pi_A = [(1, 2), (2, 3), (3, 4)]
E_pi_B = [(1, 2), (3, 4)]
E_pi_f_A = [(1, 2), (2, 3), (3, 4), (4, 5)]
E_pi_f_B = [(1, 2), (2, 3), (3, 4), (4, 5)]

# 计算精度和召回率
precision = link_precision(E_pi_A, E_pi_B)
recall = link_recall(E_pi_A, E_pi_B, E_pi_f_A, E_pi_f_B)

# 计算F1分数
f1_score = link_f1_score(precision, recall)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
