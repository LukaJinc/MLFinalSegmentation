import pandas as pd
import numpy as np
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
                    print('undersampled')
                    rus = RandomUnderSampler(sampling_strategy=0.1, random_state=RANDOM_STATE)
                    X, y = rus.fit_resample(X, y)

                #

                clf.fit(X, y)
                trees[n_clusters][cluster] = clf
        return trees

    def describe_clusters(self):
        for n_clusters in self.k_range:
            descriptions_n = {}
            total_f1_score = 0

            for cluster in range(n_clusters):
                tree = self.decision_trees[n_clusters][cluster]
                y_true = (self.final_cluster_assignments[f'cluster_{n_clusters}'] == cluster).astype(int)
                y_pred = tree.predict(self.data)

                query = self.get_cluster_query(tree, cluster)
                cluster_f1 = f1_score(y_true, y_pred, average='binary')

                descriptions_n[cluster] = {
                    'query': query,
                    'f1_score': cluster_f1
                }
                total_f1_score += cluster_f1

            self.descriptions[n_clusters] = descriptions_n
            avg_f1_score = total_f1_score / n_clusters
            print(f'<-- F1_Score for {n_clusters} clusters is {avg_f1_score:.4f} -->')

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