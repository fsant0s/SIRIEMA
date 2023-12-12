import math
import numpy as np
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    mean_squared_error,
    mean_absolute_error,
)

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.base import clone

import logging

logger = logging.getLogger(__name__)

'''
 Not using anymore. Remove it in next versions.
def calc_classification_metrics(pred_scores, pred_labels, labels):
    if len(np.unique(labels)) == 2:  # binary classification
        roc_auc_pred_score = roc_auc_score(labels, pred_scores)
        precisions, recalls, thresholds = precision_recall_curve(labels, pred_scores)
        fscore = (2 * precisions * recalls) / (precisions + recalls)
        fscore[np.isnan(fscore)] = 0
        ix = np.argmax(fscore)
        threshold = thresholds[ix].item()
        pr_auc = auc(recalls, precisions)
        tn, fp, fn, tp = confusion_matrix(labels, pred_labels, labels=[0, 1]).ravel()
        result = {
            "roc_auc": roc_auc_pred_score,
            "threshold": threshold,
            "pr_auc": pr_auc,
            "recall": recalls[ix].item(),
            "precision": precisions[ix].item(),
            "f1": fscore[ix].item(),
            "tn": tn.item(),
            "fp": fp.item(),
            "fn": fn.item(),
            "tp": tp.item(),
        }
    else:
        acc = (pred_labels == labels).mean()
        f1_micro = f1_score(y_true=labels, y_pred=pred_labels, average="micro")
        f1_macro = f1_score(y_true=labels, y_pred=pred_labels, average="macro")
        f1_weighted = f1_score(y_true=labels, y_pred=pred_labels, average="weighted")

        result = {
            "acc": acc,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "mcc": matthews_corrcoef(labels, pred_labels),
        }

    return result

def calc_regression_metrics(preds, labels):
    mse = mean_squared_error(labels, preds)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(labels, preds)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,

    }
'''

def calc_stability_metrics(data, clusters, k_means_random_state, n_samples, randomStateSeed):
    size_sample = data.shape[0]
    stabilities = {}
    
    for cluster in clusters:
        kmeans = KMeans(n_clusters=cluster, n_init=10, random_state = k_means_random_state)
        labels, indices = get_labels_and_indices(data, kmeans, size_sample, n_samples, randomStateSeed)   

        stabilities["adjusted_rand_score - " + str(cluster)] = adjusted_rand_score_func(labels, indices, cluster)
        stabilities["adjusted_mutual_info_score - " + str(cluster)] = adjusted_mutual_info_score_func(labels, indices, cluster)
    
    return stabilities

def get_labels_and_indices(data, clrt_algorithm, size_sample, n_samples, randomStateSeed):
    rng = np.random.RandomState(randomStateSeed)
    labels = []
    indices = []
    for _ in range(n_samples):
        # draw bootstrap samples, store indices
        sample_indices = rng.randint(0, data.shape[0], size_sample)
        indices.append(sample_indices)
        clrt_algorithm = clone(clrt_algorithm)
        if hasattr(clrt_algorithm, "random_state"):
            # randomize estimator if possible
            clrt_algorithm.random_state = rng.randint(1e5)
        data_bootstrap = data[sample_indices]
        clrt_algorithm.fit(data_bootstrap)
        # store clustering outcome using original indices
        relabel = -np.ones(data.shape[0], dtype=int)
        relabel[sample_indices] = clrt_algorithm.labels_
        labels.append(relabel)
    return (labels, indices)

#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
def adjusted_rand_score_func(labels, indices, cluster):
    logger.info("Computing adjusted_rand_score k = "+ str(cluster))
    scores = []
    for l, i in zip(labels, indices):
        for k, j in zip(labels, indices):
            in_both = np.intersect1d(i, j)
            scores.append(adjusted_rand_score(l[in_both], k[in_both])) 
    return np.mean(scores)

#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html
def adjusted_mutual_info_score_func(labels, indices, cluster):
    logger.info("Computing adjusted_mutual_info_score k = " + str(cluster))
    scores = []
    for l, i in zip(labels, indices):
        for k, j in zip(labels, indices):
            in_both = np.intersect1d(i, j)
            scores.append(adjusted_mutual_info_score(l[in_both], k[in_both]))
    return np.mean(scores)