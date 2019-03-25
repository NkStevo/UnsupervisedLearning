import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import time
import sys
import daz
import csv

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.feature_selection import RFE

from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

import matplotlib.cm as cm
from sklearn.datasets import load_digits

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from scipy.stats import randint as sp_randint
from sklearn.metrics import silhouette_samples, silhouette_score
from random import uniform
from pathlib import Path

def ann_analysis(neural_network, X_train, Y_train, X_test, Y_test, type):
    print("--------ANN SCORES FOR " + type + "--------")

    print("Fitting neural network...")
    start_time = time.time()
    neural_network = neural_network.fit(X_train, Y_train)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    ann_score = str(neural_network.score(X_test, Y_test))
    print("Neural Network (MLP) Accuracy using " + type + " Data: " + ann_score)

    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=35)
    plot_learning_curve(neural_network, "Neural Network (MLP) Learning Curve using " + type + " Data", X_train, Y_train, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figures/ANN/neural_network_' + type.lower() + '.png', bbox_inches='tight')
    plt.clf()

    if Path("TestData/ann_scores.csv").is_file():
        row = [type, ann_score, elapsed_time]

        with open('TestData/ann_scores.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)

        csvFile.close()
    else:
        csvData = [["Data Type", "Mean Accuracy", "Elapsed Time"], [type, ann_score, elapsed_time]]

        with open('TestData/ann_scores.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csvData)

            csvFile.close()

    print("\n")

def cluster_features(k, y_pred):
    cluster_data = []

    for label in y_pred:
        sample = [0] * k
        sample[label] = 1

        cluster_data.append(sample)

    return cluster_data


def ann_feature_curve(neural_network, X_test, X_train, Y_test, Y_train, random_state, n):
    csvData = [["Number of Features", "PCA (ANN) Accuracy", "ICA (ANN) Accuracy", "RFE (ANN) Accuracy", "GRP (ANN) Accuracy", "K-Means (ANN) Accuracy", "EM (ANN) Accuracy"]]

    for iter in range(1, n):
        row = [iter]

        pca_reduced_test = PCA(n_components=iter, random_state=random_state).fit_transform(X_test)
        pca_reduced_train = PCA(n_components=iter, random_state=random_state).fit_transform(X_train)

        ica_reduced_test = FastICA(n_components=iter, random_state=random_state).fit_transform(X_test)
        ica_reduced_train = FastICA(n_components=iter, random_state=random_state).fit_transform(X_train)

        model = LogisticRegression()
        rfe = RFE(model, iter)
        rfe_reduced_test = rfe.fit_transform(X_test, y=Y_test)
        rfe_reduced_train = rfe.fit_transform(X_train, y=Y_train)

        grp_reduced_train = GaussianRandomProjection(n_components=iter).fit_transform(X_train)
        grp_reduced_test = GaussianRandomProjection(n_components=iter).fit_transform(X_test)

        kmeans = KMeans(n_clusters=iter, random_state=random_state)
        em = GaussianMixture(n_components=iter, covariance_type='full', max_iter=200, n_init=20, random_state=random_state)

        kmeans_reduced_train = cluster_features(iter, kmeans.fit_predict(X_train))
        kmeans_reduced_test = cluster_features(iter, kmeans.fit_predict(X_test))

        em_train = em.fit(X_train)
        em_test = em.fit(X_test)

        em_reduced_train = cluster_features(iter, em_train.predict(X_train))
        em_reduced_test = cluster_features(iter, em_test.predict(X_test))

        print("Fitting neural network...")
        start_time = time.time()
        neural_network = neural_network.fit(pca_reduced_train, Y_train)
        elapsed_time = time.time() - start_time
        print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        row.append(str(neural_network.score(pca_reduced_test, Y_test)))

        print("Fitting neural network...")
        start_time = time.time()
        neural_network = neural_network.fit(ica_reduced_train, Y_train)
        elapsed_time = time.time() - start_time
        print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        row.append(str(neural_network.score(ica_reduced_test, Y_test)))

        print("Fitting neural network...")
        start_time = time.time()
        neural_network = neural_network.fit(rfe_reduced_train, Y_train)
        elapsed_time = time.time() - start_time
        print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        row.append(str(neural_network.score(rfe_reduced_test, Y_test)))

        print("Fitting neural network...")
        start_time = time.time()
        neural_network = neural_network.fit(grp_reduced_train, Y_train)
        elapsed_time = time.time() - start_time
        print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        row.append(str(neural_network.score(grp_reduced_test, Y_test)))

        print("Fitting neural network...")
        start_time = time.time()
        neural_network = neural_network.fit(kmeans_reduced_train, Y_train)
        elapsed_time = time.time() - start_time
        print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        row.append(str(neural_network.score(kmeans_reduced_test, Y_test)))

        print("Fitting neural network...")
        start_time = time.time()
        neural_network = neural_network.fit(em_reduced_train, Y_train)
        elapsed_time = time.time() - start_time
        print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        row.append(str(neural_network.score(em_reduced_test, Y_test)))

        csvData.append(row)

    with open('TestData/ann_feature_curve.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)

        csvFile.close()


def cluster_scatter(X, y_pred, title, filename):
    print("Plotting " + title + "...")
    start_time = time.time()
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title(title)
    plt.savefig(filename + "_0-1.png", bbox_inches='tight')
    plt.clf()
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("Plotting " + title + "...")
    start_time = time.time()
    plt.scatter(X[:, 1], X[:, 2], c=y_pred)
    plt.title(title)
    plt.savefig(filename + "_1-2.png", bbox_inches='tight')
    plt.clf()
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("Plotting " + title + "...")
    start_time = time.time()
    plt.scatter(X[:, 0], X[:, 2], c=y_pred)
    plt.title(title)
    plt.savefig(filename+ "_0-2.png", bbox_inches='tight')
    plt.clf()
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

def cluster_scoring(y_pred, y, type):
    print("Scoring clusters " + type + "...")
    start_time = time.time()
    ar_score = str(adjusted_rand_score(y_pred, y))
    ami_score = str(adjusted_mutual_info_score(y_pred, y))
    fm_score = str(fowlkes_mallows_score(y_pred, y))

    print("--------SCORES FOR " + type + "--------")
    print("Adjusted Rand Score for " + type + ": " + ar_score)
    print("Mutual Information Score for " + type + ": " + ami_score)
    print("Fowlkes Mallows Score for " + type + ": " + fm_score)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    print("\n")


    print("Writing to file " + type + "...")
    start_time = time.time()
    if Path("TestData/cluster_scores.csv").is_file():
        row = [type, ar_score, ami_score, fm_score]

        with open('TestData/cluster_scores.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)

        csvFile.close()
    else:
        csvData = [["Cluster Type", "Adjusted Rand Score", "Adjusted Mutual Information Score", "Fowlkes Mallows Score"], [type, ar_score, ami_score, fm_score]]

        with open('TestData/cluster_scores.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csvData)

            csvFile.close()
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

def silhouette_scoring(X, seed):

    for cluster_range in range(3, 11):
        range_n_clusters = range(2, cluster_range)

        for n_clusters in range_n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=seed)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

        plt.savefig('Figures/K-Means/Silhouette/SilhouetteAnalysis_' + str(len(range_n_clusters) + 1) + '.png', bbox_inches='tight')
        plt.clf()
        print(str(cluster_range) + " finished")

def elbow_method(X, random_state):
    error_axis = []
    k_axis = []

    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        y_pred_kmeans = kmeans.fit_predict(X)

        error_axis.append(kmeans.inertia_)
        k_axis.append(k)


    plt.plot(k_axis, error_axis)
    plt.title("K-Means Inertia")
    plt.savefig('Figures/K-Means/Inertia/inertia_plot_10.png', bbox_inches='tight')
    plt.clf()

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def main():
    df = pd.read_csv("Datasets/weatherAUS.csv", error_bad_lines=False)

    encoder = LabelEncoder()
    random_state = 325

    df.dropna()
    df.drop('Date', axis=1, inplace=True)
    df.drop('RISK_MM', axis=1, inplace=True)

    df = df[df["Location"] != "NA"]
    df = df[df["WindGustDir"] != "NA"]
    df = df[df["WindDir9am"] != "NA"]
    df = df[df["WindDir3pm"] != "NA"]
    df = df[df["RainToday"] != "NA"]
    df = df[df["RainTomorrow"] != "NA"]

    df["Location"] = encoder.fit_transform(df["Location"].astype(str))
    df["WindGustDir"] = encoder.fit_transform(df["WindGustDir"].astype(str))
    df["WindDir9am"] = encoder.fit_transform(df["WindDir9am"].astype(str))
    df["WindDir3pm"] = encoder.fit_transform(df["WindDir3pm"].astype(str))
    df["RainToday"] = encoder.fit_transform(df["RainToday"].astype(str))
    df["RainTomorrow"] = encoder.fit_transform(df["RainTomorrow"].astype(str))

    df = df[np.isfinite(df['Evaporation'])]
    df = df[np.isfinite(df['Sunshine'])]
    df = df[np.isfinite(df['WindGustSpeed'])]
    df = df[np.isfinite(df['WindSpeed9am'])]
    df = df[np.isfinite(df['WindSpeed3pm'])]
    df = df[np.isfinite(df['Humidity9am'])]
    df = df[np.isfinite(df['Humidity3pm'])]
    df = df[np.isfinite(df['Pressure9am'])]
    df = df[np.isfinite(df['Pressure3pm'])]
    df = df[np.isfinite(df['Cloud9am'])]
    df = df[np.isfinite(df['Cloud3pm'])]
    df = df[np.isfinite(df['Temp9am'])]
    df = df[np.isfinite(df['Temp3pm'])]
    df = df[np.isfinite(df['MinTemp'])]
    df = df[np.isfinite(df['MaxTemp'])]
    df = df[np.isfinite(df['Rainfall'])]

    df = df.reset_index()

    train_df, test_df = train_test_split(df, random_state=45)

    Y_train = train_df["RainTomorrow"]
    X_train = StandardScaler().fit_transform(train_df.drop("RainTomorrow", axis=1))

    Y_test = test_df["RainTomorrow"]
    X_test = StandardScaler().fit_transform(test_df.drop("RainTomorrow", axis=1))

    Y = df["RainTomorrow"]
    X = StandardScaler().fit_transform(df.drop("RainTomorrow", axis=1))

    kmeans = KMeans(n_clusters=3, random_state=random_state)
    em = GaussianMixture(n_components=3, covariance_type='full', max_iter=200, n_init=20, random_state=random_state)


    print("Fitting K-Means...")
    start_time = time.time()
    y_pred_kmeans = kmeans.fit_predict(X)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("Fitting EM...")
    start_time = time.time()
    em = em.fit(X)
    y_pred_em = em.predict(X)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    for i in range(21):
        for j in range(21):
            print("Plotting K-Means " + str(i) + "-" + str(j) + "...")
            start_time = time.time()
            plt.scatter(X[:, i], X[:, j], c=y_pred_kmeans)
            plt.title("K-Means Scatter")
            plt.savefig("Figures/K-Means/FeaturePairs/scatter_plot_" + str(i) + "_" + str(j) + ".png", bbox_inches='tight')
            plt.clf()
            elapsed_time = time.time() - start_time
            print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

            print("Plotting EM " + str(i) + "-" + str(j) + "...")
            start_time = time.time()
            plt.scatter(X[:, i], X[:, j], c=y_pred_em)
            plt.title("K-Means Scatter")
            plt.savefig("Figures/EM/FeaturePairs/scatter_plot_" + str(i) + "_" + str(j) + ".png", bbox_inches='tight')
            plt.clf()
            elapsed_time = time.time() - start_time
            print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    cluster_scoring(y_pred_kmeans, Y, "Standard K-Means Clusters")
    cluster_scoring(y_pred_em, Y, "Standard EM Clusters")

    #silhouette_scoring(X, random_state)

    print("Fitting PCA...")
    start_time = time.time()
    pca_reduced = PCA(n_components=3, random_state=random_state).fit_transform(X)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("Fitting ICA...")
    start_time = time.time()
    ica_reduced = FastICA(n_components=3, random_state=random_state).fit_transform(X)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    model = LogisticRegression()
    rfe = RFE(model, 3)

    print("Fitting RFE...")
    start_time = time.time()
    rfe_reduced = rfe.fit_transform(X, y=Y)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("Fitting PCA K-Means...")
    start_time = time.time()
    pca_pred_kmeans = kmeans.fit_predict(pca_reduced)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("Fitting PCA EM...")
    start_time = time.time()
    em = em.fit(pca_reduced)
    pca_pred_em = em.predict(pca_reduced)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


    cluster_scatter(X, pca_pred_kmeans, "PCA K-Means Clusters", 'Figures/K-Means/DReduction/pca_scatter.png')
    cluster_scatter(X, pca_pred_em, "PCA EM Clusters", 'Figures/EM/DReduction/pca_scatter.png')

    cluster_scoring(pca_pred_kmeans, Y, "PCA K-Means Clusters")
    cluster_scoring(pca_pred_em, Y, "PCA EM Clusters")

    print("Fitting ICA K-Means...")
    start_time = time.time()
    ica_pred_kmeans = kmeans.fit_predict(ica_reduced)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("Fitting ICA EM...")
    start_time = time.time()
    em = em.fit(ica_reduced)
    ica_pred_em = em.predict(ica_reduced)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    cluster_scatter(X, ica_pred_kmeans, "ICA K-Means Clusters", 'Figures/K-Means/DReduction/ica_scatter.png')
    cluster_scatter(X, ica_pred_em, "ICA EM Clusters", 'Figures/EM/DReduction/ica_scatter.png')

    cluster_scoring(ica_pred_kmeans, Y, "ICA K-Means Clusters")
    cluster_scoring(ica_pred_em, Y, "ICA EM Clusters")

    print("Fitting RFE K-Means...")
    start_time = time.time()
    rfe_pred_kmeans = kmeans.fit_predict(rfe_reduced)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("Fitting RFE EM...")
    start_time = time.time()
    em = em.fit(rfe_reduced)
    rfe_pred_em = em.predict(rfe_reduced)
    elapsed_time = time.time() - start_time
    print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    cluster_scatter(X, rfe_pred_kmeans, "RFE K-Means Clusters", 'Figures/K-Means/DReduction/rfe_scatter.png')
    cluster_scatter(X, rfe_pred_em, "RFE EM Clusters", 'Figures/EM/DReduction/rfe_scatter.png')

    cluster_scoring(rfe_pred_kmeans, Y, "RFE K-Means Clusters")
    cluster_scoring(rfe_pred_em, Y, "RFE EM Clusters")

    grp_pred_kmeans_list = []
    grp_pred_em_list = []
    grp_reduced_list = []

    for ran in range(20):
        print("Fitting GRP...")
        start_time = time.time()
        grp_reduced = GaussianRandomProjection(n_components=3).fit_transform(X)
        elapsed_time = time.time() - start_time
        print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        print("Fitting GRP K-Means...")
        start_time = time.time()
        grp_pred_kmeans = kmeans.fit_predict(grp_reduced)
        elapsed_time = time.time() - start_time
        print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        print("Fitting GRP EM...")
        em = em.fit(grp_reduced)
        grp_pred_em = em.predict(grp_reduced)
        elapsed_time = time.time() - start_time
        print("Elapsed Time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        cluster_scatter(X, grp_pred_kmeans, "GRP K-Means Clusters", 'Figures/K-Means/DReduction/RandProjections/grp_scatter_' + str(ran) + '.png')
        cluster_scatter(X, grp_pred_em, "GRP EM Clusters", 'Figures/EM/DReduction/grp_scatter_' + str(ran) + '.png')

        grp_pred_kmeans_list.append(grp_pred_kmeans)
        grp_pred_em_list.append(grp_pred_em)
        grp_reduced_list.append(grp_reduced)

        cluster_scoring(grp_pred_kmeans, Y, "GRP K-Means Clusters " + str(ran))
        cluster_scoring(grp_pred_em, Y, "GRP EM Clusters " + str(ran))



if __name__ == '__main__':
    main()
