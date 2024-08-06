import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.tree import plot_tree, DecisionTreeClassifier

import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score

from constants import RANDOM_STATE
from utils import *


class ClusterDescriptions:
    def __init__(self, data, df_pca, identifiers, k_range):
        self.data = data
        self.k_range = k_range

        self.final_cluster_assignments = get_final_assignments(df_pca, identifiers, k_range)
        self.segment_distributions = get_segment_distributions(self.final_cluster_assignments, k_range)

        self.decision_trees = self.get_trees(data)

        self.descriptions = {}

    def plot_segment_traffic(self):
        print('Segment Traffic')

        for n_clusters in self.k_range:
            print(f"*** Segment Traffic for {n_clusters} Clusters ***")
            print('Cluster Distribution')
            print(self.final_cluster_assignments[f'cluster_{n_clusters}'].value_counts())

            SEGMENT = [x for x in range(n_clusters)]

            for cluster in SEGMENT:
                if (n_clusters - 1) in self.k_range:
                    print(f'From Previous Segments Segment {cluster} Took')
                    for index, value in self.segment_distributions[n_clusters][cluster].items():
                        percent = round(value, 2)
                        if percent == 0:
                            continue
                        print(f'{percent} from Segment {index}')

    def get_trees(self, data):
        trees = {}
        for n_clusters in self.k_range:
            tree_n = {}
            SEGMENT = [x for x in range(n_clusters)]

            for cluster in SEGMENT:
                # clf = DecisionTreeClassifier(max_depth = 3, min_samples_leaf=1000, min_impurity_decrease=0.01, random_state=RANDOM_STATE)
                clf = DecisionTreeClassifier(max_depth=3, min_impurity_decrease=0.01, random_state=RANDOM_STATE)
                X = data
                y = (self.final_cluster_assignments[f'cluster_{n_clusters}'] == cluster).astype(int)

                if y.mean() < 0.09 and y.sum() > 1000:
                    rus = RandomUnderSampler(sampling_strategy=0.1, random_state=RANDOM_STATE)
                    X, y = rus.fit_resample(X, y)

                clf.fit(X, y)

                tree_n[cluster] = clf

            trees[n_clusters] = tree_n
        return trees

    def get_max_described_leaf(self, n_clusters, cluster):
        tr = self.decision_trees[n_clusters][cluster]

        n_nodes = tr.tree_.node_count
        children_left = tr.tree_.children_left
        children_right = tr.tree_.children_right
        weighted_n_node_samples = tr.tree_.weighted_n_node_samples
        value = tr.tree_.value

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

        return is_leaves, max_described_i

    def gen(self, path, n_clusters, cluster, features, is_leaves):
        tr = self.decision_trees[n_clusters][cluster]

        n_nodes = tr.tree_.node_count
        children_left = tr.tree_.children_left
        feature = tr.tree_.feature
        threshold = tr.tree_.threshold

        res = ""
        sign = ""
        for i in range(n_nodes):
            if path[i] == 1 and is_leaves[i] != 1:
                if path[children_left[i]] == 1:
                    sign = "<"
                else:
                    sign = ">"
                res += features[feature[i]] + " " + sign + " " + str(threshold[i]) + ". "
        return res

    def describe_clusters(self, data):
        for n_clusters in self.k_range:
            f1_score = 0
            descriptions_n = {}

            SEGMENT = [x for x in range(n_clusters)]
            for cluster in SEGMENT:
                desc = {}
                tr = self.decision_trees[n_clusters][cluster]
                y = (self.final_cluster_assignments[f'cluster_{n_clusters}'] == cluster).astype(int)

                is_leaves, max_described_i = self.get_max_described_leaf(n_clusters, cluster)

                if max_described_i == -1:
                    print('Could not Explain This Segment With Less than 4 Features')

                    desc['query'] = 'Could not Explain This Segment With Less than 4 Features'
                    desc['f1_score'] = 0
                    descriptions_n[cluster] = desc

                    continue

                # Generate Paths To Leaf Nodes That Explains Most of Current Segment
                leaf_id = pd.DataFrame(tr.apply(data))

                sample_ind = leaf_id[leaf_id[0] == max_described_i].index[0]
                path = tr.decision_path(data[sample_ind:sample_ind + 1]).toarray()
                query = self.gen(path[0], n_clusters, cluster, list(data.columns), is_leaves)

                weighted_n_node_samples = tr.tree_.weighted_n_node_samples
                value = tr.tree_.value

                precision = weighted_n_node_samples[max_described_i] * value[max_described_i][0][1] / \
                            weighted_n_node_samples[max_described_i]
                recall = weighted_n_node_samples[max_described_i] * value[max_described_i][0][1] / y.sum()

                f1_score_n = 2 * ((precision * recall) / (precision + recall))

                print('f1_score', f1_score_n)

                f1_score += f1_score_n

                desc['query'] = query
                desc['f1_score'] = f1_score_n
                descriptions_n[cluster] = desc

            print(f'<-- F1_Score for {n_clusters} clusters is {f1_score / n_clusters} -->')
            self.descriptions[n_clusters] = descriptions_n

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
                  max_depth=3,  # Limit depth for readability
                  proportion=True,
                  precision=2,
                  impurity=False)
        plt.title(f"Decision Tree for {n_clusters} clusters, Cluster {cluster}")
        plt.tight_layout()
        plt.savefig(f"./decision_tree/cluster_{n_clusters}_tree_{cluster}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_all_trees(self):
        for n_clusters in self.k_range:
            for cluster in range(n_clusters):
                self.plot_decision_tree(n_clusters, cluster)
