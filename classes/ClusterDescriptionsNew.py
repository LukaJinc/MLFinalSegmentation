import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from constants import RANDOM_STATE
from utils import get_final_assignments, get_segment_distributions, calculate_roc_auc_scores
from imblearn.under_sampling import RandomUnderSampler


def get_max_described_leaf(tree):
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


class ClusterDescriptionsNew:
    def __init__(self, data, df_pca, identifiers, k_range, test_size=0.2):
        self.k_range = k_range

        # Split data and df_pca into train and test sets
        self.data_train, self.data_test, self.df_pca_train, self.df_pca_test = train_test_split(
            data, df_pca, test_size=test_size, random_state=RANDOM_STATE)

        self.k_means, self.final_cluster_assignments = get_final_assignments(self.df_pca_train, identifiers,
                                                                             k_range)
        self.segment_distributions = get_segment_distributions(self.final_cluster_assignments, k_range)
        self.decision_trees = self.get_trees()
        self.descriptions = {}

    def get_trees(self):
        trees = {}
        for n_clusters in self.k_range:
            trees[n_clusters] = {}
            for cluster in range(n_clusters):
                clf = DecisionTreeClassifier(max_depth=3, min_impurity_decrease=0.01, random_state=RANDOM_STATE)
                X = self.data_train
                y = (self.final_cluster_assignments[f'cluster_{n_clusters}'] == cluster).astype(int)

                #

                if y.mean() < 0.09 and y.sum() > 1000:
                    rus = RandomUnderSampler(sampling_strategy=0.1, random_state=RANDOM_STATE)
                    X, y = rus.fit_resample(X, y)

                #

                clf.fit(X, y)
                trees[n_clusters][cluster] = clf
        return trees

    def describe_clusters(self):
        for n_clusters in self.k_range:
            descriptions_n = {}
            cluster_metrics = []
            cluster_instances = []
            y_true = self.final_cluster_assignments[f'cluster_{n_clusters}']

            y_pred = np.zeros_like(y_true)
            y_pred_proba = np.zeros((len(y_true), n_clusters))

            for cluster in range(n_clusters):
                tree = self.decision_trees[n_clusters][cluster]
                y_cluster = (y_true == cluster).astype(int)
                cluster_instances.append(y_cluster.sum())

                max_described_leaf = get_max_described_leaf(tree)
                leaf_id = tree.apply(self.data_train)
                y_pred_cluster = (leaf_id == max_described_leaf).astype(int)
                y_pred[y_pred_cluster == 1] = cluster

                # Get probabilities for ROC AUC calculation
                y_pred_proba[:, cluster] = tree.predict_proba(self.data_train)[:, 1]

                query = self.get_cluster_query(tree)
                cluster_f1 = f1_score(y_cluster, y_pred_cluster, average='binary')
                cluster_metrics.append({'f1': cluster_f1})

                descriptions_n[cluster] = {
                    'query': query,
                    'metrics': {'f1': cluster_f1}
                }

            # Calculate train ROC AUC scores
            train_macro_roc_auc, train_weighted_roc_auc, train_per_cluster_roc_auc = calculate_roc_auc_scores(
                y_true, y_pred_proba, n_clusters
            )

            # Update cluster metrics and descriptions with ROC AUC scores
            for cluster in range(n_clusters):
                cluster_metrics[cluster]['roc_auc'] = train_per_cluster_roc_auc[cluster]
                descriptions_n[cluster]['metrics']['roc_auc'] = train_per_cluster_roc_auc[cluster]

            train_weighted_f1 = np.average([m['f1'] for m in cluster_metrics], weights=cluster_instances)
            train_macro_f1 = np.average([m['f1'] for m in cluster_metrics])

            # Evaluate on test set
            test_results = self.evaluate(self.data_test, self.df_pca_test, n_clusters)

            self.descriptions[n_clusters] = descriptions_n

            print(f'<-- Results for {n_clusters} clusters -->')
            print(f'Train F1-score: Weighted {train_weighted_f1:.4f}, Macro {train_macro_f1:.4f}')
            print(f'Train ROC AUC: Weighted {train_weighted_roc_auc:.4f}, Macro {train_macro_roc_auc:.4f}')
            print(f'Test F1-score: Weighted {test_results["weighted_f1"]:.4f}, Macro {test_results["macro_f1"]:.4f}')
            print(
                f'Test ROC AUC: Weighted {test_results["weighted_roc_auc"]:.4f}, Macro {test_results["macro_roc_auc"]:.4f}')
            print()

    def get_cluster_query(self, tree):
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right

        def dfs(node, depth=0):
            if children_left[node] == children_right[node]:
                return node, ""

            if tree.tree_.value[children_left[node]][0][1] > tree.tree_.value[children_right[node]][0][1]:
                best_child, child_query = dfs(children_left[node], depth + 1)
                return best_child, f"{self.data_train.columns[feature[node]]} <= {threshold[node]:.6f}. " + child_query
            else:
                best_child, child_query = dfs(children_right[node], depth + 1)
                return best_child, f"{self.data_train.columns[feature[node]]} > {threshold[node]:.6f}. " + child_query

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
                  feature_names=self.data_train.columns,
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

    def evaluate(self, test_data, test_pca, n_clusters):
        cluster_metrics = []
        cluster_instances = []

        if n_clusters not in self.decision_trees:
            raise ValueError(f"No decision tree found for {n_clusters} clusters")

        kmeans_model = self.k_means[n_clusters]
        test_labels = kmeans_model.predict(test_pca)

        y_pred = np.zeros_like(test_labels)
        y_pred_proba = np.zeros((len(test_labels), n_clusters))

        for cluster in range(n_clusters):
            tree = self.decision_trees[n_clusters][cluster]
            y_cluster = (test_labels == cluster).astype(int)
            cluster_instances.append(y_cluster.sum())

            max_described_leaf = get_max_described_leaf(tree)
            leaf_id = tree.apply(test_data)

            y_pred_cluster = (leaf_id == max_described_leaf).astype(int)
            y_pred[y_pred_cluster == 1] = cluster

            # Get probabilities for ROC AUC calculation
            y_pred_proba[:, cluster] = tree.predict_proba(test_data)[:, 1]

            cluster_f1 = f1_score(y_cluster, y_pred_cluster, average='binary')
            cluster_metrics.append({'f1': cluster_f1})

        # Calculate ROC AUC scores
        macro_roc_auc, weighted_roc_auc, per_cluster_roc_auc = calculate_roc_auc_scores(
            test_labels, y_pred_proba, n_clusters
        )

        # Update cluster metrics with ROC AUC scores
        for cluster in range(n_clusters):
            cluster_metrics[cluster]['roc_auc'] = per_cluster_roc_auc[cluster]

        weighted_f1 = np.average([m['f1'] for m in cluster_metrics], weights=cluster_instances)
        macro_f1 = np.average([m['f1'] for m in cluster_metrics])

        return {
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'macro_roc_auc': macro_roc_auc,
            'weighted_roc_auc': weighted_roc_auc,
            'per_cluster_metrics': cluster_metrics,
            'test_labels': test_labels
        }


