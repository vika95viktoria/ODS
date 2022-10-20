import numpy as np
import math


def entropy(a_list):
    total = len(a_list)
    ent = 0
    unique, counts = np.unique(a_list, return_counts=True)
    c_dict = dict(zip(unique, counts))
    for x in set(a_list):
        count_x = c_dict[x]
        ent = ent - (count_x / total) * math.log((count_x / total), 2)
    return ent


def information_gain(root, left, right):
    s_0 = entropy(root)
    s_1 = entropy(left)
    s_2 = entropy(right)
    return s_0 - (len(left) / len(root)) * s_1 - (len(right) / len(root)) * s_2


def best_feature_to_split(X, y):
    '''Outputs information gain when splitting on best feature'''
    num_of_features = X.shape[1]
    num_of_samples = X.shape[0]
    if num_of_samples == 1 or entropy(y) == 0:
        return
    full_matrix = np.hstack([X, y.reshape(-1, 1)])
    max_information_gain = 0
    best_feature_num = -1
    split_index = -1
    for f in range(num_of_features):
        sorted_matrix = full_matrix[full_matrix[:, f].argsort()]
        sorted_y = sorted_matrix[:, num_of_features]
        y_ind = 1
        current_y = sorted_matrix[0][num_of_features]
        while y_ind < num_of_samples:
            while y_ind < num_of_samples and sorted_matrix[y_ind][num_of_features] == current_y:
                current_y = sorted_matrix[y_ind][num_of_features]
                y_ind += 1
            if y_ind < num_of_samples:
                info_gain = information_gain(sorted_y, sorted_y[:y_ind], sorted_y[y_ind:])
                if info_gain > max_information_gain:
                    max_information_gain = info_gain
                    best_feature_num = f
                    split_index = y_ind
                current_y = sorted_matrix[y_ind][num_of_features]
                y_ind += 1
    print(f"Current Max information gain is {max_information_gain}, best feature {best_feature_num}, index for split {split_index}")
    print(f"Final feature to split: {best_feature_num}, index: {split_index}")
    sorted_matrix_final = full_matrix[full_matrix[:, best_feature_num].argsort()]
    sorted_y_final = sorted_matrix_final[:, num_of_features]
    left, left_y = sorted_matrix_final[:split_index, :num_of_features], sorted_y_final[:split_index]
    right, right_y = sorted_matrix_final[split_index:, :num_of_features], sorted_y_final[split_index:]
    best_feature_to_split(left, left_y)
    best_feature_to_split(right, right_y)


size = 10

X = np.hstack([np.random.normal(size=size).reshape(-1, 1),
               np.random.normal(loc=5, size=size).reshape(-1, 1),
               np.random.normal(loc=25, size=size).reshape(-1, 1)])
y = np.random.randint(low=0, high=2, size=(size,))

best_feature_to_split(X, y)
