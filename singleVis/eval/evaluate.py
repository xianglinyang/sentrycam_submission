"""
Help functions to evaluate our visualization system
"""

import numpy as np
import torch.nn as nn
import torch.optim as optim
from pynndescent import NNDescent
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness
from scipy.stats import spearmanr, pearsonr, rankdata


def evaluate_proj_nn_perseverance_knn(data, embedding, n_neighbors, metric="euclidean"):
    """
    evaluate projection function, nn preserving property using knn algorithm
    :param data: ndarray, high dimensional representations
    :param embedding: ndarray, low dimensional representations
    :param n_neighbors: int, the number of neighbors
    :param metric: str, by default "euclidean"
    :return nn property: float, nn preserving property
    """
    n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20.0))
    n_iters = max(5, int(round(np.log2(data.shape[0]))))
    # get nearest neighbors
    nnd = NNDescent(
        data,
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True
    )
    high_ind, _ = nnd.neighbor_graph
    nnd = NNDescent(
        embedding,
        n_neighbors=n_neighbors,
        metric="euclidean",
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True
    )
    low_ind, _ = nnd.neighbor_graph

    border_pres = np.zeros(len(data))
    for i in range(len(data)):
        border_pres[i] = len(np.intersect1d(high_ind[i], low_ind[i]))

    # return border_pres.mean(), border_pres.max(), border_pres.min()
    return border_pres.mean()


def evaluate_proj_nn_perseverance_trustworthiness(data, embedding, n_neighbors, metric="euclidean"):
    """
    evaluate projection function, nn preserving property using trustworthiness formula
    :param data: ndarray, high dimensional representations
    :param embedding: ndarray, low dimensional representations
    :param n_neighbors: int, the number of neighbors
    :param metric: str, by default "euclidean"
    :return nn property: float, nn preserving property
    """
    t = trustworthiness(data, embedding, n_neighbors=n_neighbors, metric=metric)
    return t


def evaluate_proj_boundary_perseverance_knn(data, embedding, high_centers, low_centers, n_neighbors, metric):
    """
    evaluate projection function, boundary preserving property
    :param data: ndarray, high dimensional representations
    :param embedding: ndarray, low dimensional representations
    :param high_centers: ndarray, border points high dimensional representations
    :param low_centers: ndarray, border points low dimensional representations
    :param n_neighbors: int, the number of neighbors
    :return boundary preserving property: float,boundary preserving property
    """
    high_neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=0.4, metric=metric)
    high_neigh.fit(high_centers)
    high_ind = high_neigh.kneighbors(data, n_neighbors=n_neighbors, return_distance=False)

    low_neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=0.4, metric=metric)
    low_neigh.fit(low_centers)
    low_ind = low_neigh.kneighbors(embedding, n_neighbors=n_neighbors, return_distance=False)

    border_pres = np.zeros(len(data))
    for i in range(len(data)):
        border_pres[i] = len(np.intersect1d(high_ind[i], low_ind[i]))

    # return border_pres.mean(), border_pres.max(), border_pres.min()
    return border_pres.mean()


def evaluate_proj_temporal_perseverance_corr(alpha, delta_x):
    """
    Evaluate temporal preserving property,
    calculate the correlation between neighbor preserving rate and moving distance in low dim in a time sequence
    :param alpha: ndarray, shape(N,) neighbor preserving rate
    :param delta_x: ndarray, shape(N,), moved distance in low dim for each point
    :return corr: ndarray, shape(N,), correlation for each point from temporal point of view
    """
    alpha = alpha.T
    delta_x = delta_x.T
    shape = alpha.shape
    data_num = shape[0]
    corr = np.zeros(data_num)
    for i in range(data_num):
        # correlation, pvalue = spearmanr(alpha[:, i], delta_x[:, i])
        correlation, pvalue = pearsonr(alpha[i], delta_x[i])
        if np.isnan(correlation):
            correlation = 0.0
        corr[i] = correlation
    return corr.mean(), corr.std()


def evaluate_inv_distance(data, inv_data):
    """
    The distance between original data and reconstruction data
    :param data: ndarray, high dimensional data
    :param inv_data: ndarray, reconstruction data
    :return err: mse, reconstruction error
    """
    return np.linalg.norm(data-inv_data, axis=1).mean()
    # if metric == "euclidean":
    #     return np.linalg.norm(data-inv_data, axis=1)
    # elif metric == "cosine":
    #     return np.array([1 - np.dot(data[i], inv_data[i])/np.linalg.norm(data[i])*np.linalg.norm(inv_data[i]) for i in range(len(data))])
    # else:
    #     raise NotImplementedError

def evaluate_embedding_distance(source, target, metric, one_target):
    if metric == "euclidean":
        return np.linalg.norm(source-target, axis=1)
    elif metric == "cosine":
        if one_target:
            return np.array([1 - np.dot(source[i], target)/np.linalg.norm(source[i])*np.linalg.norm(target) for i in range(len(source))])
        else:
            return np.array([1 - np.dot(source[i], target[i])/np.linalg.norm(source[i])*np.linalg.norm(target[i]) for i in range(len(source))])
    else:
        raise NotImplementedError


def evaluate_inv_accu(labels, pred):
    """
    prediction accuracy of reconstruction data
    :param labels: ndarray, shape(N,), label for each point
    :param pred: ndarray, shape(N,), prediction for each point
    :return accu: float, the reconstruction accuracy
    """
    return np.sum(labels == pred) / len(labels)


def evaluate_inv_conf(labels, ori_pred, new_pred):
    """
    the confidence difference between original data and reconstruction data
    :param labels: ndarray, shape(N,), the original prediction for each point
    :param ori_pred: ndarray, shape(N,10), the prediction of original data
    :param new_pred: ndarray, shape(N,10), the prediction of reconstruction data
    :return diff: float, the mean of confidence difference for each point
    """
    old_conf = [ori_pred[i, labels[i]] for i in range(len(labels))]
    new_conf = [new_pred[i, labels[i]] for i in range(len(labels))]
    old_conf = np.array(old_conf)
    new_conf = np.array(new_conf)

    diff = np.abs(old_conf - new_conf)
    # return diff.mean(), diff.max(), diff.min()
    return diff.mean()


def evaluate_proj_temporal_perseverance_entropy(alpha, delta_x):
    """
    (discard)
    calculate the temporal preserving property
    based on the correlation between the entropy of moved distance and neighbor preserving rate(alpha)
    :param alpha: ndarray, shape(N,), neighbor preserving rate for each point
    :param delta_x: ndarray, shape(N,), the moved distance in low dim for each point
    :return corr: float, the mean of all correlation
    """
    alpha = alpha.T
    delta_x = delta_x.T
    shape = alpha.shape
    # normalize
    # delta_x_norm = delta_x.max(-1)
    # delta_x_norm = (delta_x.T/delta_x_norm).T
    delta_x_norm = delta_x.max()
    delta_x_norm = delta_x / delta_x_norm

    alpha = np.floor(alpha*10)
    delta_x_norm = np.floor(delta_x_norm*10)

    corr = np.zeros(len(alpha))
    # samples
    for i in range(len(alpha)):
        # alpha0-alpha9
        index = list()
        entropy = list()
        for j in range(11):
            dx = delta_x_norm[i][np.where(alpha[i] == j)]
            entropy_x = np.zeros(11)
            for k in range(11):
                entropy_x[k] = np.sum(dx == k)
            if np.sum(entropy_x) == 0:
                continue
            else:
                entropy_x = entropy_x / np.sum(entropy_x+10e-8)
                entropy_x = np.sum(entropy_x*np.log(entropy_x+10e-8))
                entropy.append(-entropy_x)
                index.append(j)
        if len(index) < 2:
            print("no enough data to form a correlation, setting correlation to be 0")
            corr[i] = 0
        else:
            correlation, _ = pearsonr(index, entropy)
            corr[i] = correlation

    return corr.mean()


def gradient_diff(prev_e, next_e, training_data, targets, data_provider, criterion):

    model_t = data_provider.model_function(prev_e)
    model_t = model_t.to(data_provider.DEVICE)
    optimizer_t = optim.SGD(model_t.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    model_t1 = data_provider.model_function(next_e)
    model_t1 = model_t1.to(data_provider.DEVICE)
    optimizer_t1 = optim.SGD(model_t1.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    num = len(training_data)
    gradient_diff = np.zeros(num)
    for i in range(num):
        x = training_data[i:i+1]
        target = targets[i:i+1]

        # Forward pass and compute gradients at time t
        output_t = model_t(x)
        loss_t = criterion(output_t, target)
        optimizer_t.zero_grad()
        loss_t.backward()

        # Save gradients at time t
        grads_t = [p.grad.clone() for p in model_t.parameters()]

        # Forward pass and compute gradients at time t+1
        output_t1 = model_t1(x)
        loss_t1 = criterion(output_t1, target)
        optimizer_t1.zero_grad()
        loss_t1.backward()

        # Save gradients at time t+1
        grads_t1 = [p.grad.clone() for p in model_t1.parameters()]

        # Compute cosine similarity between gradients at t and t+1
        cos_sim_values = []
        cos = nn.CosineSimilarity(dim=0)
        for g_t, g_t1 in zip(grads_t, grads_t1):
            cos_sim = cos(g_t.flatten(), g_t1.flatten())
            cos_sim_values.append(cos_sim.item())

        # Average cosine similarity
        avg_cos_sim = sum(cos_sim_values) / len(cos_sim_values)

        gradient_diff[i] = 1 - avg_cos_sim

    return gradient_diff


def evaluate_proj_temporal_global_corr(high_rank, low_rank):
    l = len(high_rank)
    tau_l = np.zeros(l)
    p_l = np.zeros(l)
    for i in range(l):
        r1 = high_rank[i]
        r2 = low_rank[i]
        tau, p = spearmanr(r1, r2)
        tau_l[i] = tau
        p_l[i] = p
    return tau_l, p_l


def _wcov(x, y, w, ms):
    return np.sum(w * (x - ms[0]) * (y - ms[1]))
def _wpearson(x, y, w):
    mx, my = (np.sum(i * w) / np.sum(w) for i in [x, y])
    return _wcov(x, y, w, [mx, my]) / np.sqrt(_wcov(x, x, w, [mx, mx]) * _wcov(y, y, w, [my, my]))
def evaluate_proj_temporal_weighted_global_corr(high_rank, low_rank):
    k = len(high_rank)
    r = rankdata(high_rank).astype("int")-1
    tau = _wpearson(high_rank[r], low_rank[r], 1/np.arange(1, k+1))
    return tau


def evaluate_keep_B(low_B, grid_view, decision_view, metric, threshold=0.8):
    """
    evaluate whether high dimensional boundary points still lying on Boundary in low-dimensional space or not
    find the nearest grid point of boundary points, and check whether the color of corresponding grid point is white or not

    :param low_B: ndarray, (n, 2), low dimension position of boundary points
    :param grid_view: ndarray, (resolution^2, 2), the position array of grid points
    :param decision_view: ndarray, (resolution^2, 3), the RGB color of grid points
    :param threshold:
    :return:
    """
    if len(low_B) == 0 or low_B is None:
        return .0
    # reshape grid and decision view
    grid_view = grid_view.reshape(-1, 2)
    decision_view = decision_view.reshape(-1, 3)

    # find the color of nearest grid view
    nbs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", metric=metric).fit(grid_view)
    _, indices = nbs.kneighbors(low_B)
    indices = indices.squeeze()
    sample_colors = decision_view[indices]

    # check whether 3 channel are above a predefined threshold
    c1 = np.zeros(indices.shape[0], dtype=np.bool)
    c1[sample_colors[:, 0] > threshold] = 1

    c2 = np.zeros(indices.shape[0], dtype=np.bool)
    c2[sample_colors[:, 1] > threshold] = 1

    c3 = np.zeros(indices.shape[0], dtype=np.bool)
    c3[sample_colors[:, 2] > threshold] = 1
    c = np.logical_and(c1, c2)
    c = np.logical_and(c, c3)

    # return the ratio of boundary points that still lie on boundary after dimension reduction
    return np.sum(c)/len(c)