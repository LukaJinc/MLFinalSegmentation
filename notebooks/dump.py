# def generate_description(path, n_nodes, features, is_leaves, children_left, feature, threshold):
#     res = ""
#     sign = ""
#     for i in range(n_nodes):
#         if path[i] == 1 and is_leaves[i] != 1:
#             if path[children_left[i]] == 1:
#                 sign = "<"
#             else:
#                 sign = ">"
#             res += features[feature[i]] + " " + sign + " " + str(threshold[i]) + ". "
#     return res
#
#
# def cluster_descriptions(data, df_pca, identifiers, k_range):
#     print(f'Whole population are {data.shape[0]} clients.')
#     final_cluster_assignments = get_final_assignments(df_pca, identifiers, k_range)
#     segment_distributions = get_segment_distributions(final_cluster_assignments, k_range)
#
#     f1_scores = pd.Series()
#
#     for n_clusters in k_range:
#         print(f"*** Cluster Description for {n_clusters} Clusters ***")
#         SEGMENT = [x for x in range(n_clusters)]
#
#         print('Cluster Distribution')
#         print(final_cluster_assignments[f'cluster_{n_clusters}'].value_counts())
#
#         k_f1_score = 0
#         for cluster in SEGMENT:
#             # clf = DecisionTreeClassifier(max_depth = 3, min_samples_leaf=1000, min_impurity_decrease=0.01, random_state=RANDOM_STATE)
#             clf = DecisionTreeClassifier(max_depth=3, min_impurity_decrease=0.01, random_state=RANDOM_STATE)
#
#             X = data
#             y = (final_cluster_assignments[f'cluster_{n_clusters}'] == cluster).astype(int)
#
#             print(f"  Segment - {cluster} contains {y.sum()} clients, that are {y.mean()} percent of whole population.")
#
#             if (n_clusters - 1) in k_range:
#                 print(f'From Previous Segments Segment {cluster} Took')
#                 for index, value in segment_distributions[n_clusters][cluster].items():
#                     percent = round(value, 2)
#                     if percent == 0:
#                         continue
#                     print(f'{percent} from Segment {index}')
#
#             if y.mean() < 0.09 and y.sum() > 1000:
#                 rus = RandomUnderSampler(sampling_strategy=0.1, random_state=RANDOM_STATE)
#                 X, y = rus.fit_resample(X, y)
#                 print('target undersampled')
#
#             clf.fit(X, y)
#
#             n_nodes = clf.tree_.node_count
#             children_left = clf.tree_.children_left
#             children_right = clf.tree_.children_right
#             feature = clf.tree_.feature
#             threshold = clf.tree_.threshold
#             weighted_n_node_samples = clf.tree_.weighted_n_node_samples
#             value = clf.tree_.value
#
#             is_leaves = (children_left == children_right)
#             classes = np.zeros(n_nodes)
#             for i in range(n_nodes):
#                 if is_leaves[i] != 0:
#                     classes[i] = value[i][0][1] > value[i][0][0]
#
#             found_points = {}
#
#             for i in range(n_nodes):
#                 if classes[i] == 1:
#                     found_points.update({i: weighted_n_node_samples[i] * value[i][0][1]})
#
#             max_described_num = 0
#             max_described_i = -1
#             for key in found_points:
#                 if found_points[key] > max_described_num:
#                     max_described_num = found_points[key]
#                     max_described_i = key
#
#             if max_described_i == -1:
#                 print('Could not Explain This Segment With Less than 4 Features')
#                 continue
#             # Generate Paths To Leaf Nodes That Explains Most of Current Segment
#
#             paths = {}
#             leaf_id = pd.DataFrame(clf.apply(X))
#
#             sample_ind = leaf_id[leaf_id[0] == max_described_i].index[0]
#             path = clf.decision_path(X[sample_ind:sample_ind + 1]).toarray()
#             paths.update({max_described_i: path})
#
#             # Generate Description
#             for i in paths.keys():
#                 print('** Description **')
#
#                 print(generate_description(paths[i][0], n_nodes, list(data.columns), is_leaves, children_left, feature,
#                                            threshold))
#                 print(
#                     f'--------- In given population {weighted_n_node_samples[i]} clients satisfies this filter. From this people, {weighted_n_node_samples[i] * value[i][0][1]} are in Segment {cluster}')
#
#                 precision = weighted_n_node_samples[i] * value[i][0][1] / weighted_n_node_samples[i]
#                 recall = weighted_n_node_samples[i] * value[i][0][1] / y.sum()
#                 f1_score = 2 * ((precision * recall) / (precision + recall))
#
#                 k_f1_score += f1_score
#                 print(f'Precision - {precision}, Recall - {recall}')
#                 print(f'F1_Score {f1_score}')
#
#         print(f'<-- F1_Score for {n_clusters} clusters is {k_f1_score / n_clusters} -->')
#         f1_scores[n_clusters] = k_f1_score / n_clusters
#     print(f1_scores)