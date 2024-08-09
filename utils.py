import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

from constants import RANDOM_STATE


def get_final_assignments(df_pca, identifiers, k_range):
    final_cluster_assignments = pd.DataFrame(columns=['ID'], index=df_pca.index)
    final_cluster_assignments['ID'] = identifiers
    k_means = {}
    for n_clusters in k_range:
        best_kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=10, random_state=RANDOM_STATE)
        final_cluster_assignments[f'cluster_{n_clusters}'] = best_kmeans.fit_predict(df_pca)
        k_means[n_clusters] = best_kmeans
    return k_means, final_cluster_assignments


def get_segment_distributions(final_cluster_assignments, k_range):
    distributions = {}
    for k in k_range:
        if (k - 1) not in k_range:
            continue
        dist = {}
        vs = pd.DataFrame(final_cluster_assignments.groupby(f'cluster_{k}')[f'cluster_{k - 1}'].value_counts(
            normalize=True)).reset_index()
        for segment in range(k):
            temp = vs.loc[vs[f'cluster_{k}'] == segment, [f'cluster_{k - 1}', 'proportion']]
            temp.index = temp[f'cluster_{k - 1}']
            dist.update({segment: temp['proportion']})
        distributions.update({k: dist})
    return distributions


# interquantile search
def detect_outliers_iqr(df):
    outliers = pd.DataFrame()
    for column in df.select_dtypes(include='number').columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR

        outlier_condition = (df[column] < lower_fence) | (df[column] > upper_fence)
        outliers[column] = (outlier_condition).astype(int)

    return outliers.sum(axis=1)


# data - population with features, columnID - column name that is of identifier column
def pca_data(data, columnID, n_components, drop_outliers=False):
    identifiers = data.pop(columnID)

    s = StandardScaler()
    scaled_data = s.fit_transform(data)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca_data = pca.fit_transform(scaled_data)
    df_pca = pd.DataFrame(pca_data)

    variance_explained = pca.explained_variance_ratio_
    cumulative_variance_explained = np.sum(variance_explained)

    print('PCA Result - ', cumulative_variance_explained)

    # Drops outliers from pca_data using interquantile search
    if drop_outliers:
        pca_count = detect_outliers_iqr(df_pca)
        threshold = pca_count.quantile(0.99)

        print(f'Outlier is point if it is outlier in {threshold} Features')

        identifiers = identifiers[pca_count[pca_count <= threshold].index].reset_index(drop=True)
        df_pca = df_pca.loc[pca_count[pca_count <= threshold].index, :].reset_index(drop=True)
        data = data.loc[pca_count[pca_count <= threshold].index, :].reset_index(drop=True)

    return data, identifiers, df_pca


# fills Features that have 0 in it with 0, otherwise with median
def fill_na_median(cc_df):
    vs = pd.DataFrame({'name': cc_df.columns[1:]})
    vs['min'] = vs['name'].apply(lambda x: round(cc_df[x].min(), 2))
    vs['median'] = vs['name'].apply(lambda x: round(cc_df[x].median(), 2))

    for name in list(vs.loc[vs['min'] != 0, 'name'].values):
        cc_df[name] = cc_df[name].fillna(cc_df[name].median())

    for name in list(vs.loc[vs['min'] == 0, 'name'].values):
        cc_df[name] = cc_df[name].fillna(cc_df[name].min())


def cluster_in_range(df_pca, identifiers, k_range):
    wcss = []
    diff_clusters = {}

    for n_clusters in k_range:
        kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=10, random_state=RANDOM_STATE)

        cluster_assignments = pd.DataFrame(kmeans.fit_predict(df_pca), columns=['cluster'], index=df_pca.index)
        cluster_assignments['ID'] = identifiers

        diff_clusters[n_clusters] = cluster_assignments['cluster'].value_counts()

        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    plt.plot(k_range, wcss, marker='o')

    plt.xticks(k_range)
    plt.grid()
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    # plt.savefig('num_clust.png')
    plt.show()

    return diff_clusters
