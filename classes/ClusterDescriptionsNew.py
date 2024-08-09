import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from constants import RANDOM_STATE
from utils import get_final_assignments, get_segment_distributions
from imblearn.under_sampling import RandomUnderSampler


class ClusterDescriptionsNew:
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
            trees[n_clusters] = {}
            for cluster in range(n_clusters):
                clf = DecisionTreeClassifier(max_depth=3, min_impurity_decrease=0.01, random_state=RANDOM_STATE)
                X = self.data
                y = (self.final_cluster_assignments[f'cluster_{n_clusters}'] == cluster).astype(int)

                #

                if y.mean() < 0.09 and y.sum() > 1000:
                    rus = RandomUnderSampler(sampling_strategy=0.1, random_state=RANDOM_STATE)
                    X, y = rus.fit_resample(X, y)

                #

                clf.fit(X, y)
                trees[n_clusters][cluster] = clf
        return trees

    def get_max_described_leaf(self, tree):
        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        weighted_n_node_samples = tree.tree_.weighted_n_node_samples
        value = tree.tree_.value

        is_leaves = (children_left == children_right)
        classes = np.zeros(n_nodes)
        for i in range(n_nodes):
            if is_leaves[i] != 0:
                classes[i] = value[i][0][1] > value[i][0][0]

        found_points = {}

        for i in range(n_nodes):
            if classes[i] == 1:
                found_points.update({i: weighted_n_node_samples[i] * value[i][0][1]})

        max_described_num = 0
        max_described_i = -1
        for key in found_points:
            if found_points[key] > max_described_num:
                max_described_num = found_points[key]
                max_described_i = key

        return max_described_i

    def describe_clusters(self, f1_type='weighted'):
        if f1_type not in ['binary', 'weighted']:
            raise ValueError("f1_type must be either 'binary' or 'weighted'")

        for n_clusters in self.k_range:
            descriptions_n = {}
            cluster_f1_scores = []
            cluster_instances = []
            y_true = self.final_cluster_assignments[f'cluster_{n_clusters}']

            for cluster in range(n_clusters):
                tree = self.decision_trees[n_clusters][cluster]
                y_cluster = (y_true == cluster).astype(int)
                cluster_instances.append(y_cluster.sum())

                max_described_leaf = self.get_max_described_leaf(tree)
                leaf_id = tree.apply(self.data)
                y_pred = (leaf_id == max_described_leaf).astype(int)

                query = self.get_cluster_query(tree, max_described_leaf)
                cluster_f1 = f1_score(y_cluster, y_pred, average='binary')
                cluster_f1_scores.append(cluster_f1)

                descriptions_n[cluster] = {
                    'query': query,
                    'f1_score': cluster_f1
                }

            if f1_type == 'binary':
                f1_score_n = np.average(cluster_f1_scores)
            else:
                f1_score_n = np.average(cluster_f1_scores, weights=cluster_instances)
            self.descriptions[n_clusters] = descriptions_n

            print(f'<-- Weighted F1_Score for {n_clusters} clusters is {f1_score_n:.4f} -->')

            # for cluster in range(n_clusters):
            #     print(
            #         f'Cluster {cluster}: F1-score = {cluster_f1_scores[cluster]:.4f}, Instances = {cluster_instances[cluster]}')


    def get_cluster_query(self, tree, cluster):
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right

        def dfs(node, depth=0):
            if children_left[node] == children_right[node]:
                return node, ""

            if tree.tree_.value[children_left[node]][0][1] > tree.tree_.value[children_right[node]][0][1]:
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

    def plot_decision_tree(self, n_clusters, cluster):
        if n_clusters not in self.decision_trees or cluster not in self.decision_trees[n_clusters]:
            raise ValueError(f"No decision tree found for {n_clusters} clusters and cluster {cluster}")

        tree = self.decision_trees[n_clusters][cluster]

        plt.figure(figsize=(20, 10))
        plot_tree(tree,
                  feature_names=self.data.columns,
                  class_names=['Other', f'Cluster {cluster}'],
                  filled=True,
                  rounded=True,
                  max_depth=3,
                  proportion=True,
                  precision=2,
                  impurity=False)
        plt.title(f"New Decision Tree for {n_clusters} clusters, Cluster {cluster}")
        plt.tight_layout()
        plt.savefig(f"./new_decision_tree/cluster_{n_clusters}_tree_{cluster}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_all_trees(self):
        for n_clusters in self.k_range:
            for cluster in range(n_clusters):
                self.plot_decision_tree(n_clusters, cluster)

    def get_descriptions(self, n_clusters):
        if n_clusters not in self.descriptions:
            raise ValueError(f"Descriptions for {n_clusters} clusters have not been generated yet.")
        return self.descriptions[n_clusters]
