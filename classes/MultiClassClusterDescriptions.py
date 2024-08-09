import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score, roc_auc_score
from constants import RANDOM_STATE
from utils import get_final_assignments, get_segment_distributions
import matplotlib.pyplot as plt


class MultiClassClusterDescriptions:
    def __init__(self, data, df_pca, identifiers, k_range):
        self.k_range = k_range
        self.data = data
        self.final_cluster_assignments = get_final_assignments(df_pca, identifiers, k_range)
        self.segment_distributions = get_segment_distributions(self.final_cluster_assignments, k_range)
        self.decision_trees = self.get_trees()
        self.descriptions = {}

    def get_trees(self):
        trees = {}
        for n_clusters in self.k_range:
            clf = DecisionTreeClassifier(max_depth=3, min_impurity_decrease=0.01, random_state=RANDOM_STATE)
            X = self.data
            y = self.final_cluster_assignments[f'cluster_{n_clusters}']
            clf.fit(X, y)
            trees[n_clusters] = clf
        return trees

    def get_max_described_leaf(self, tree, cluster):
        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        weighted_n_node_samples = tree.tree_.weighted_n_node_samples
        value = tree.tree_.value

        is_leaves = (children_left == children_right)
        classes = np.zeros(n_nodes)
        for i in range(n_nodes):
            if is_leaves[i] != 0:
                classes[i] = np.argmax(value[i][0])

        found_points = {}
        for i in range(n_nodes):
            if classes[i] == cluster:
                found_points.update({i: weighted_n_node_samples[i] * value[i][0][cluster]})

        max_described_num = 0
        max_described_i = -1
        for key in found_points:
            if found_points[key] > max_described_num:
                max_described_num = found_points[key]
                max_described_i = key

        return max_described_i

    def describe_clusters(self, f1_type):
        if f1_type not in ['binary', 'weighted']:
            raise ValueError("f1_type must be either 'binary' or 'weighted'")

        for n_clusters in self.k_range:
            descriptions_n = {}
            cluster_f1_scores = []
            cluster_instances = []

            tree = self.decision_trees[n_clusters]
            y_true = self.final_cluster_assignments[f'cluster_{n_clusters}']

            # Use max_described_leaf for predictions
            y_pred = np.zeros_like(y_true)
            for cluster in range(n_clusters):
                max_described_leaf = self.get_max_described_leaf(tree, cluster)
                leaf_id = tree.apply(self.data)
                y_pred[leaf_id == max_described_leaf] = cluster

            for cluster in range(n_clusters):
                query = self.get_cluster_query(tree, cluster)
                cluster_instances.append(np.sum(y_true == cluster))
                cluster_f1 = f1_score(y_true == cluster, y_pred == cluster, average='binary')
                cluster_f1_scores.append(cluster_f1)

                descriptions_n[cluster] = {
                    'query': query,
                    'f1_score': cluster_f1
                }

            self.descriptions[n_clusters] = descriptions_n

            if f1_type == 'binary':
                f1_score_n = np.average(cluster_f1_scores)
            else:
                f1_score_n = f1_score(y_true, y_pred, average='weighted')

            # print(f'<-- Descriptions generated for {n_clusters} clusters -->')
            print(f'<-- Weighted F1-score for {n_clusters} clusters: {f1_score_n:.4f} -->')

            # for cluster in range(n_clusters):
            #     print(
            #         f'Cluster {cluster}: F1-score = {cluster_f1_scores[cluster]:.4f}, Instances = {cluster_instances[cluster]}')
            #
            # print()

    def get_cluster_query(self, tree, cluster):
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        value = tree.tree_.value
        n_classes = tree.n_classes_

        def dfs(node, depth=0):
            if children_left[node] == children_right[node]:
                return node, ""

            left_samples = value[children_left[node]].reshape(-1)
            right_samples = value[children_right[node]].reshape(-1)

            if len(left_samples) < n_classes:
                left_samples = np.pad(left_samples, (0, n_classes - len(left_samples)))
            if len(right_samples) < n_classes:
                right_samples = np.pad(right_samples, (0, n_classes - len(right_samples)))

            left_ratio = left_samples[cluster] / left_samples.sum() if left_samples.sum() > 0 else 0
            right_ratio = right_samples[cluster] / right_samples.sum() if right_samples.sum() > 0 else 0

            if left_ratio > right_ratio:
                best_child, child_query = dfs(children_left[node], depth + 1)
                return best_child, f"{self.data.columns[feature[node]]} <= {threshold[node]:.6f}. " + child_query
            else:
                best_child, child_query = dfs(children_right[node], depth + 1)
                return best_child, f"{self.data.columns[feature[node]]} > {threshold[node]:.6f}. " + child_query

        best_leaf, query = dfs(0)
        return query.strip()

    def plot_segment_traffic(self):
        print('Segment Traffic')
        for n_clusters in self.k_range:
            print(f"*** Segment Traffic for {n_clusters} Clusters ***")
            print('Cluster Distribution')
            print(self.final_cluster_assignments[f'cluster_{n_clusters}'].value_counts())

            if n_clusters > 1 and (n_clusters - 1) in self.k_range:
                for cluster in range(n_clusters):
                    print(f'Segment {cluster} Composition:')
                    for index, value in self.segment_distributions[n_clusters][cluster].items():
                        percent = round(value, 2)
                        if percent > 0:
                            print(f'{percent}% from Previous Segment {index}')

    def get_descriptions(self, n_clusters):
        if n_clusters not in self.descriptions:
            raise ValueError(f"Descriptions for {n_clusters} clusters have not been generated yet.")
        return self.descriptions[n_clusters]

    def plot_decision_tree(self, n_clusters):
        if n_clusters not in self.decision_trees:
            raise ValueError(f"No decision tree found for {n_clusters} clusters")

        tree = self.decision_trees[n_clusters]

        plt.figure(figsize=(20, 10))
        plot_tree(tree,
                  feature_names=self.data.columns,
                  class_names=[f'Cluster {i}' for i in range(n_clusters)],
                  filled=True,
                  rounded=True,
                  max_depth=3,  # Limit depth for readability
                  proportion=True,
                  precision=2,
                  impurity=False)
        plt.title(f"Decision Tree for {n_clusters} clusters")
        plt.tight_layout()
        plt.savefig(f"./multi_class_tree/multiclass_tree_{n_clusters}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_all_trees(self):
        for n_clusters in self.k_range:
            self.plot_decision_tree(n_clusters)
