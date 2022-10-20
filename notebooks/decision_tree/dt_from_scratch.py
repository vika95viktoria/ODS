from math import log2
import numpy as np
from sklearn.base import BaseEstimator
from collections import Counter
from sklearn.datasets import load_digits

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split


def get_probabilities(y):
    elem_map = {}
    n = len(y)
    for y_elem in y:
        if y_elem in elem_map:
            elem_map[y_elem] = elem_map[y_elem] + 1
        else:
            elem_map[y_elem] = 1
    return [x / n for x in elem_map.values()]


def entropy(y):
    probabilities = get_probabilities(y)
    return -sum([p * log2(p) for p in probabilities])


def gini(y):
    probabilities = get_probabilities(y)
    return 1 - sum([p * p for p in probabilities])


def variance(y):
    y_mean = np.mean(y)
    n = len(y)
    return sum([(y_elem - y_mean) ** 2 for y_elem in y]) / n


def mad_median(y):
    y_median = np.median(y)
    n = len(y)
    return sum([abs(y_elem - y_median) for y_elem in y]) / n


class Node:
    def __init__(self, feature_idx=0, threshold=0, labels=None, left=None, right=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.labels = labels
        self.left = left
        self.right = right


def regression_leaf(y):
    return np.mean(y)


def classification_leaf(y):
    data = Counter(y)
    return data.most_common(1)[0][0]


class DecisionTree(BaseEstimator):
    method_map = {'gini': gini, 'entropy': entropy, 'variance': variance, 'mad_median': mad_median}

    def get_method(self, method_name):
        return self.method_map.get(method_name, gini)

    def get_leaf_method(self, method_name):
        if method_name in ['gini', 'entropy']:
            return classification_leaf
        else:
            return regression_leaf

    def __init__(
            self, max_depth=np.inf, min_samples_split=2, criterion="gini", debug=False
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = self.get_method(criterion)
        self.debug = debug
        self.leaf_predictor = self.get_leaf_method(criterion)
        self.tree = None

    def information_gain(self, X, X_left, X_right):
        return self.criterion(X) - (len(X_left) / len(X)) * self.criterion(X_left) - (
                len(X_right) / len(X)) * self.criterion(X_right)

    def best_feature_to_split(self, X, y):
        '''Outputs information gain when splitting on best feature'''
        num_of_features = X.shape[1]
        num_of_samples = X.shape[0]
        if num_of_samples == 1 or self.criterion(y) == 0:
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
                    info_gain = self.information_gain(sorted_y, sorted_y[:y_ind], sorted_y[y_ind:])
                    if info_gain > max_information_gain:
                        max_information_gain = info_gain
                        best_feature_num = f
                        split_index = y_ind
                    current_y = sorted_matrix[y_ind][num_of_features]
                    y_ind += 1
        print(f"Final feature to split: {best_feature_num}, index: {split_index}")
        sorted_matrix_final = full_matrix[full_matrix[:, best_feature_num].argsort()]
        sorted_y_final = sorted_matrix_final[:, num_of_features]
        left, left_y = sorted_matrix_final[:split_index, :num_of_features], sorted_y_final[:split_index]
        right, right_y = sorted_matrix_final[split_index:, :num_of_features], sorted_y_final[split_index:]
        threshold = (sorted_matrix_final[split_index - 1, best_feature_num] + sorted_matrix_final[
            split_index, best_feature_num]) / 2
        return Node(feature_idx=best_feature_num, threshold=threshold, labels=y,
                    left=self.best_feature_to_split(left, left_y), right=self.best_feature_to_split(right, right_y))

    def fit(self, X, y):
        self.tree = self.best_feature_to_split(X, y)
        return self.tree

    def predict(self, X):
        y_pred = []
        for x in X:
            current_node = self.tree
            while current_node.left is not None:
                if x[current_node.feature_idx] < current_node.threshold:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
            y_pred.append(self.get_leaf_method(current_node.labels))
        return y_pred

    def predict_proba(self, X):
        y_pred = []
        for x in X:
            current_node = self.tree
            while current_node.left is not None:
                if x[current_node.feature_idx] < current_node.threshold:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
            count = Counter(current_node.labels)
            proba = []
            n = len(current_node.labels)
            for c in count.keys():
                proba.append(count[c] / n)
            y_pred.append(proba)
        return y_pred


digits = load_digits()
X = digits['data']
y = digits['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
dt = DecisionTree()
dt.fit(X_train, y_train)
dt.predict(X_test)
