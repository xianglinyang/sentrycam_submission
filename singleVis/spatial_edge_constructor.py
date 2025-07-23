from abc import ABC, abstractmethod

import numpy as np
import os
import time
import math
import json
import scipy

from umap.umap_ import fuzzy_simplicial_set, make_epochs_per_sample, smooth_knn_dist, compute_membership_strengths
from pynndescent import NNDescent
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from singleVis.kcenter_greedy import kCenterGreedy
from singleVis.intrinsic_dim import IntrinsicDim
from singleVis.subsampling import *
from singleVis.backend import get_graph_elements, get_attention
from singleVis.utils import find_neighbor_preserving_rate

def normalize_data(train_data):
    train_max = train_data.max(axis=0)
    train_min = train_data.min(axis=0)
    train_data = (train_data - train_max)/(train_max-train_min)
    return train_data

# @numba.njit(
# locals={
#     "knn_dists": numba.types.float32[:, ::1],
#     "val": numba.types.float32,
# },
# parallel=True,
# fastmath=True,
# )
def cosine_fuzzy_set(knn_indices, knn_dists, n_neighbors):
    # construct a fuzzy_simplicial_set
    knn_dists = knn_dists.astype(np.float32)
    n_samples = knn_indices.shape[0]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  
            if knn_indices[i, j] == i:
                val = 0.0
            else:
                val = (1 - knn_dists[i][j])/2

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val
    return rows, cols, vals

class SpatialEdgeConstructorAbstractClass(ABC):
    @abstractmethod
    def __init__(self, data_provider) -> None:
        pass

    @abstractmethod
    def construct(self, *args, **kwargs):
        # return head, tail, weight, feature_vectors
        pass

    @abstractmethod
    def record_time(self, save_dir, file_name, operation, t):
        pass

'''Base class for Spatial Edge Constructor'''
class SpatialEdgeConstructor(SpatialEdgeConstructorAbstractClass):
    '''Construct spatial complex
    '''
    def __init__(self, data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors, metric) -> None:
        """Init parameters for spatial edge constructor

        Parameters
        ----------
        data_provider : data.DataProvider
             data provider
        init_num : int
            init number to calculate c
        s_n_epochs : int
            the number of epochs to fit for one iteration(epoch)
            e.g. n_epochs=5 means each edge will be sampled 5*prob times in one training epoch
        b_n_epochs : int
            the number of epochs to fit boundary samples for one iteration (epoch)
        n_neighbors: int
            local connectivity
        """
        self.data_provider = data_provider
        self.init_num = init_num
        self.s_n_epochs = s_n_epochs
        self.b_n_epochs = b_n_epochs
        self.n_neighbors = n_neighbors
        self.metric = metric
    
    def _construct_fuzzy_complex(self, train_data):
        """
        construct a vietoris-rips complex
        """
        # number of trees in random projection forest
        n_trees = min(64, 5 + int(round((train_data.shape[0]) ** 0.5 / 20.0)))
        # max number of nearest neighbor iters to perform
        n_iters = max(5, int(round(np.log2(train_data.shape[0]))))
        # get nearest neighbors
        nnd = NNDescent(
            train_data,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True
        )
        knn_indices, knn_dists = nnd.neighbor_graph
        random_state = check_random_state(None)
        complex, sigmas, rhos = fuzzy_simplicial_set(
            X=train_data,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            random_state=random_state,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )
        return complex, sigmas, rhos, knn_indices
    
    def _construct_boundary_wise_complex(self, train_data, border_centers, train_sigmas=None, train_rhos=None):
        """compute the boundary wise complex
            for each border point, we calculate its k nearest train points
            for each train data, we calculate its k nearest border points
        """
        high_neigh = NearestNeighbors(n_neighbors=self.n_neighbors, radius=0.4)
        high_neigh.fit(border_centers)
        fitting_data = np.concatenate((train_data, border_centers), axis=0)
        knn_dists, knn_indices = high_neigh.kneighbors(fitting_data, n_neighbors=self.n_neighbors, return_distance=True)
        knn_indices = knn_indices + len(train_data)

        # random_state = check_random_state(None)
        # bw_complex, sigmas, rhos = fuzzy_simplicial_set(
        #     X=fitting_data,
        #     n_neighbors=self.n_neighbors,
        #     metric=self.metric,
        #     random_state=random_state,
        #     knn_indices=knn_indices,
        #     knn_dists=knn_dists,
        # )
        knn_dists = knn_dists.astype(np.float32)

        sigmas, rhos = smooth_knn_dist(
            knn_dists,
            float(self.n_neighbors),
            local_connectivity=1.0,
        )
        if train_sigmas is not None:
            sigmas[:len(train_data)] = train_sigmas
        if train_rhos is not None:
            rhos[:len(train_data)] = train_rhos

        rows, cols, vals, _ = compute_membership_strengths(
            knn_indices, knn_dists, sigmas, rhos
        )
        bw_complex = scipy.sparse.coo_matrix(
            (vals, (rows, cols)), shape=(fitting_data.shape[0], fitting_data.shape[0])
        )
        bw_complex.eliminate_zeros()
        transpose = bw_complex.transpose()
        prod_matrix = bw_complex.multiply(transpose)
        set_op_mix_ratio = 1.0
        bw_complex = (
            set_op_mix_ratio * (bw_complex + transpose - prod_matrix)
            + (1.0 - set_op_mix_ratio) * prod_matrix
        )

        bw_complex.eliminate_zeros()

        return bw_complex, sigmas, rhos, knn_indices
    
    def _construct_local_temporal_complex(self, prev_data, next_data, b_num=0):
        """
        construct a temporal complex
        """
        n_neighbors = 5
        
        # cosine distance
        high_neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=0.4, metric="cosine")
        high_neigh.fit(prev_data)
        knn_dists, knn_indices = high_neigh.kneighbors(next_data, n_neighbors=n_neighbors, return_distance=True)
        knn_indices = knn_indices + len(next_data) + b_num

        # ignore the structure of prev_data
        knn_dists = np.concatenate((knn_dists, np.zeros((len(prev_data) + b_num, n_neighbors))), axis=0)
        knn_indices = np.concatenate((knn_indices, -np.ones((len(prev_data) + b_num, n_neighbors))), axis=0)

        rows, cols, vals = cosine_fuzzy_set(knn_indices, knn_dists, n_neighbors)

        result = scipy.sparse.coo_matrix(
            (vals, (rows, cols)), shape=(len(prev_data)+len(next_data)+b_num, len(prev_data)+len(next_data)+b_num)
        )
        result.eliminate_zeros()

        transpose = result.transpose()
        prod_matrix = result.multiply(transpose)
        set_op_mix_ratio = 1.0
        result = (
            set_op_mix_ratio * (result + transpose - prod_matrix)
            + (1.0 - set_op_mix_ratio) * prod_matrix
        )

        result.eliminate_zeros()

        return result

    
    def _construct_step_edge_dataset(self, vr_complex, bw_complex, t_complex=None):
        """
        construct the mixed edge dataset for one time step
            connect border points and train data(both direction)
        :param vr_complex: Vietoris-Rips complex
        :param bw_complex: boundary-augmented complex
        :param t_complex: temporal-wise complex
        :param n_epochs: the number of epoch that we iterate each round
        :return: edge dataset
        """
        # TODO weight alignment
        # get data from graph

        if vr_complex is None:
            head = None
            tail = None
            weight = None
        else:
            _, head, tail, weight, _ = get_graph_elements(vr_complex, self.s_n_epochs)
        
        # get data from graph
        if bw_complex is not None and self.b_n_epochs > 0:
            if head is None:
                _, head, tail, weight, _ = get_graph_elements(bw_complex, self.b_n_epochs)
            else:
                _, bw_head, bw_tail, bw_weight, _ = get_graph_elements(bw_complex, self.b_n_epochs)
                head = np.concatenate((head, bw_head), axis=0)
                tail = np.concatenate((tail, bw_tail), axis=0)
                weight = np.concatenate((weight, bw_weight), axis=0)
        
        if t_complex is not None:
            if head is None:
                _, head, tail, weight, _ = get_graph_elements(t_complex, self.t_n_epochs)
            else:
                _, t_head, t_tail, t_weight, _ = get_graph_elements(t_complex, self.t_n_epochs)
                head = np.concatenate((head, t_head), axis=0)
                tail = np.concatenate((tail, t_tail), axis=0)
                weight = np.concatenate((weight, t_weight), axis=0)
        
        if head is None:
            print("Not edge avaiable!")
        return head, tail, weight
    

    def construct(self):
        return NotImplemented
    
    def record_time(self, save_dir, file_name, operation, t):
        file_path = os.path.join(save_dir, file_name+".json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                ti = json.load(f)
        else:
            ti = dict()
        ti[operation] = t
        with open(file_path, "w") as f:
            json.dump(ti, f)
        
'''
Strategies:
    Random: random select samples
    KC: select coreset using k center greedy algorithm (recommend)
    KC Parallel: parallel selecting samples
    KC Hybrid: additional term for repley connecting epochs
'''

class RandomSpatialEdgeConstructor(SpatialEdgeConstructor):
    def __init__(self, data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors, metric, ratio) -> None:
        super().__init__(data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors, metric)
        self.ratio = ratio
    
    def construct(self):
        # dummy input
        edge_to = None
        edge_from = None
        sigmas = None
        rhos = None
        weight = None
        probs = None
        feature_vectors = None
        attention = None
        knn_indices = None
        time_step_nums = list()
        time_step_idxs_list = list()

        train_num = self.data_provider.train_num
        curr_selected_idxs = np.arange(train_num)

        # each time step
        for t in range(self.data_provider.s, self.data_provider.e+1, self.data_provider.p):
            # load train data and border centers
            train_data = self.data_provider.train_representation(t).squeeze()

            rs = RandomSampling(train_data[curr_selected_idxs], metric="euclidean")
            selected_idxs_tmp = rs.sampling(self.ratio)
            curr_selected_idxs = curr_selected_idxs[selected_idxs_tmp]

            train_data = train_data[curr_selected_idxs]
            time_step_idxs_list.append(curr_selected_idxs.tolist())

            if self.b_n_epochs != 0:
                border_centers = self.data_provider.border_representation(t).squeeze()
                border_centers = border_centers
                complex, sigmas_t1, rhos_t1, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                bw_complex, sigmas_t2, rhos_t2, _ = self._construct_boundary_wise_complex(train_data, border_centers)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, bw_complex)
                sigmas_t = np.concatenate((sigmas_t1, sigmas_t2[len(sigmas_t1):]), axis=0)
                rhos_t = np.concatenate((rhos_t1, rhos_t2[len(rhos_t1):]), axis=0)
                fitting_data = np.concatenate((train_data, border_centers), axis=0)
                pred_model = self.data_provider.prediction_function(t)
                attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
                t_num = len(train_data)
                b_num = len(border_centers)
            else:
                complex, sigmas_t, rhos_t, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, None, self.n_epochs)
                fitting_data = np.copy(train_data)
                pred_model = self.data_provider.prediction_function(t)
                attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
                t_num = len(train_data)
                b_num = 0

            if edge_to is None:
                edge_to = edge_to_t
                edge_from = edge_from_t
                weight = weight_t
                probs = weight_t / weight_t.max()
                feature_vectors = fitting_data
                attention = attention_t
                sigmas = sigmas_t
                rhos = rhos_t
                knn_indices = knn_idxs_t
                time_step_nums.append((t_num, b_num))
            else:
                # every round, we need to add len(data) to edge_to(as well as edge_from) index
                increase_idx = len(feature_vectors)
                edge_to = np.concatenate((edge_to, edge_to_t + increase_idx), axis=0)
                edge_from = np.concatenate((edge_from, edge_from_t + increase_idx), axis=0)
                # normalize weight to be in range (0, 1)
                weight = np.concatenate((weight, weight_t), axis=0)
                probs_t = weight_t / weight_t.max()
                probs = np.concatenate((probs, probs_t), axis=0)
                sigmas = np.concatenate((sigmas, sigmas_t), axis=0)
                rhos = np.concatenate((rhos, rhos_t), axis=0)
                feature_vectors = np.concatenate((feature_vectors, fitting_data), axis=0)
                attention = np.concatenate((attention, attention_t), axis=0)
                knn_indices = np.concatenate((knn_indices, knn_idxs_t+increase_idx), axis=0)
                time_step_nums.append((t_num, b_num))

        return edge_to, edge_from, weight, feature_vectors, time_step_nums, time_step_idxs_list, knn_indices , sigmas, rhos, attention
    

class kcSpatialEdgeConstructor(SpatialEdgeConstructor):
    def __init__(self, data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors, metric, MAX_HAUSDORFF, ALPHA, BETA, init_idxs=None, adding_num=100) -> None:
        super().__init__(data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors, metric)
        self.MAX_HAUSDORFF = MAX_HAUSDORFF
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.init_idxs = init_idxs
        self.adding_num = adding_num
    
    def _get_unit(self, data, init_num, adding_num=100):
        # normalize
        t0 = time.time()
        l = len(data)
        idxs = np.random.choice(np.arange(l), size=init_num, replace=False)
        # _,_ = hausdorff_dist_cus(data, idxs)

        id = IntrinsicDim(data)
        d0 = id.twonn_dimension_fast()
        # d0 = twonn_dimension_fast(data)

        kc = kCenterGreedy(data)
        _ = kc.select_batch_with_budgets(idxs, adding_num)
        c0 = kc.hausdorff()
        t1 = time.time()
        return c0, d0, "{:.1f}".format(t1-t0)
    
    def construct(self):
        """construct spatio-temporal complex and get edges

        Returns
        -------
        _type_
            _description_
        """

        # dummy input
        edge_to = None
        edge_from = None
        sigmas = None
        rhos = None
        weight = None
        probs = None
        feature_vectors = None
        attention = None
        knn_indices = None
        time_step_nums = list()
        time_step_idxs_list = list()

        train_num = self.data_provider.train_num
        if self.init_idxs is None:
            selected_idxs = np.random.choice(np.arange(train_num), size=self.init_num, replace=False)
        else:
            selected_idxs = np.copy(self.init_idxs)

        baseline_data = self.data_provider.train_representation(self.data_provider.e)
        max_x = np.linalg.norm(baseline_data, axis=1).max()
        baseline_data = baseline_data/max_x
        
        c0,d0,_ = self._get_unit(baseline_data, self.init_num, self.adding_num)

        if self.MAX_HAUSDORFF is None:
            self.MAX_HAUSDORFF = c0-0.01

        # each time step
        for t in range(self.data_provider.e, self.data_provider.s - 1, -self.data_provider.p):
            print("=================+++={:d}=+++================".format(t))
            # load train data and border centers
            train_data = self.data_provider.train_representation(t)

            # normalize data by max ||x||_2
            max_x = np.linalg.norm(train_data, axis=1).max()
            train_data = train_data/max_x

            # get normalization parameters for different epochs
            c,d,_ = self._get_unit(train_data, self.init_num,self.adding_num)
            c_c0 = math.pow(c/c0, self.BETA)
            d_d0 = math.pow(d/d0, self.ALPHA)
            print("Finish calculating normaling factor")

            kc = kCenterGreedy(train_data)
            _ = kc.select_batch_with_cn(selected_idxs, self.MAX_HAUSDORFF, c_c0, d_d0, p=0.95)
            selected_idxs = kc.already_selected.astype("int")

            save_dir = os.path.join(self.data_provider.content_path, "selected_idxs")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            with open(os.path.join(save_dir,"selected_{}.json".format(t)), "w") as f:
                json.dump(selected_idxs.tolist(), f)
            print("select {:d} points".format(len(selected_idxs)))

            time_step_idxs_list.insert(0, np.arange(len(selected_idxs)).tolist())

            train_data = self.data_provider.train_representation(t).squeeze()
            train_data = train_data[selected_idxs]

            if self.b_n_epochs != 0:
                # select highly used border centers...
                border_centers = self.data_provider.border_representation(t)
                t_num = len(selected_idxs)
                b_num = len(border_centers)

                complex, sigmas_t1, rhos_t1, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                bw_complex, sigmas_t2, rhos_t2, _ = self._construct_boundary_wise_complex(train_data, border_centers)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, bw_complex)
                sigmas_t = np.concatenate((sigmas_t1, sigmas_t2[len(sigmas_t1):]), axis=0)
                rhos_t = np.concatenate((rhos_t1, rhos_t2[len(rhos_t1):]), axis=0)
                fitting_data = np.concatenate((train_data, border_centers), axis=0)
                # pred_model = self.data_provider.prediction_function(t)
                # attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
                attention_t = np.ones(fitting_data.shape)
            else:
                t_num = len(selected_idxs)
                b_num = 0

                complex, sigmas_t, rhos_t, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, None)
                fitting_data = np.copy(train_data)
                # pred_model = self.data_provider.prediction_function(t)
                # attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
                attention_t = np.ones(fitting_data.shape)


            if edge_to is None:
                edge_to = edge_to_t
                edge_from = edge_from_t
                weight = weight_t
                probs = weight_t / weight_t.max()
                feature_vectors = fitting_data
                attention = attention_t
                sigmas = sigmas_t
                rhos = rhos_t
                knn_indices = knn_idxs_t
                # npr = npr_t
                time_step_nums.insert(0, (t_num, b_num))
            else:
                # every round, we need to add len(data) to edge_to(as well as edge_from) index
                increase_idx = len(fitting_data)
                edge_to = np.concatenate((edge_to_t, edge_to + increase_idx), axis=0)
                edge_from = np.concatenate((edge_from_t, edge_from + increase_idx), axis=0)
                # normalize weight to be in range (0, 1)
                weight = np.concatenate((weight_t, weight), axis=0)
                probs_t = weight_t / weight_t.max()
                probs = np.concatenate((probs_t, probs), axis=0)
                sigmas = np.concatenate((sigmas_t, sigmas), axis=0)
                rhos = np.concatenate((rhos_t, rhos), axis=0)
                feature_vectors = np.concatenate((fitting_data, feature_vectors), axis=0)
                attention = np.concatenate((attention_t, attention), axis=0)
                knn_indices = np.concatenate((knn_idxs_t, knn_indices+increase_idx), axis=0)
                # npr = np.concatenate((npr_t, npr), axis=0)
                time_step_nums.insert(0, (t_num, b_num))

        return edge_to, edge_from, weight, feature_vectors, time_step_nums, time_step_idxs_list, knn_indices, sigmas, rhos, attention


class kcParallelSpatialEdgeConstructor(SpatialEdgeConstructor):
    def __init__(self, data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors, metric, MAX_HAUSDORFF, ALPHA, BETA) -> None:
        super().__init__(data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors, metric)
        self.MAX_HAUSDORFF = MAX_HAUSDORFF
        self.ALPHA = ALPHA
        self.BETA = BETA
    
    def _get_unit(self, data, adding_num=100):
        t0 = time.time()
        l = len(data)
        idxs = np.random.choice(np.arange(l), size=self.init_num, replace=False)

        id = IntrinsicDim(data)
        d0 = id.twonn_dimension_fast()

        kc = kCenterGreedy(data)
        _ = kc.select_batch_with_budgets(idxs, adding_num)
        c0 = kc.hausdorff()
        t1 = time.time()
        return c0, d0, "{:.1f}".format(t1-t0)
    
    def construct(self):
        """construct spatio-temporal complex and get edges

        Returns
        -------
        _type_
            _description_
        """

        # dummy input
        edge_to = None
        edge_from = None
        sigmas = None
        rhos = None
        weight = None
        probs = None
        feature_vectors = None
        attention = None
        knn_indices = None
        time_step_nums = list()
        time_step_idxs_list = list()# the list of selected idxs

        train_num = self.data_provider.train_num
        init_selected_idxs = np.random.choice(np.arange(train_num), size=self.init_num, replace=False)

        baseline_data = self.data_provider.train_representation(self.data_provider.e)
        baseline_data = baseline_data.reshape(len(baseline_data), -1)
        max_x = np.linalg.norm(baseline_data, axis=1).max()
        baseline_data = baseline_data/max_x
        
        c0,d0,_ = self._get_unit(baseline_data)

        # each time step
        for t in range(self.data_provider.e, self.data_provider.s - 1, -self.data_provider.p):
            print("=================+++={:d}=+++================".format(t))
            # load train data and border centers
            train_data = self.data_provider.train_representation(t)
            train_data = train_data.reshape(len(train_data), -1)

            # normalize data by max ||x||_2
            max_x = np.linalg.norm(train_data, axis=1).max()
            train_data = train_data/max_x

            # get normalization parameters for different epochs
            c,d,_ = self._get_unit(train_data)
            c_c0 = math.pow(c/c0, self.BETA)
            d_d0 = math.pow(d/d0, self.ALPHA)
            print("Finish calculating normaling factor")

            kc = kCenterGreedy(train_data)
            _ = kc.select_batch_with_cn(init_selected_idxs, self.MAX_HAUSDORFF, c_c0, d_d0, p=0.95)
            selected_idxs = kc.already_selected.astype("int")

            save_dir = os.path.join(self.data_provider.content_path, "selected_idxs")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            with open(os.path.join(save_dir,"selected_{}.json".format(t)), "w") as f:
                json.dump(selected_idxs.tolist(), f)
            print("select {:d} points".format(len(selected_idxs)))

            time_step_idxs_list.insert(0, selected_idxs)

            train_data = self.data_provider.train_representation(t)
            train_data = train_data[selected_idxs]
            
            if self.b_n_epochs != 0:
                # select highly used border centers...
                border_centers = self.data_provider.border_representation(t).squeeze()
                t_num = len(selected_idxs)
                b_num = len(border_centers)

                complex, sigmas_t1, rhos_t1, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                bw_complex, sigmas_t2, rhos_t2, _ = self._construct_boundary_wise_complex(train_data, border_centers)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, bw_complex)
                sigmas_t = np.concatenate((sigmas_t1, sigmas_t2[len(sigmas_t1):]), axis=0)
                rhos_t = np.concatenate((rhos_t1, rhos_t2[len(rhos_t1):]), axis=0)
                fitting_data = np.concatenate((train_data, border_centers), axis=0)
                pred_model = self.data_provider.prediction_function(t)
                attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
            else:
                t_num = len(selected_idxs)
                b_num = 0

                complex, sigmas_t, rhos_t, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, None)
                fitting_data = np.copy(train_data)
                pred_model = self.data_provider.prediction_function(t)
                attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)

            if edge_to is None:
                edge_to = edge_to_t
                edge_from = edge_from_t
                weight = weight_t
                probs = weight_t / weight_t.max()
                feature_vectors = fitting_data
                attention = attention_t
                sigmas = sigmas_t
                rhos = rhos_t
                knn_indices = knn_idxs_t
                # npr = npr_t
                time_step_nums.insert(0, (t_num, b_num))
            else:
                # every round, we need to add len(data) to edge_to(as well as edge_from) index
                increase_idx = len(fitting_data)
                edge_to = np.concatenate((edge_to_t, edge_to + increase_idx), axis=0)
                edge_from = np.concatenate((edge_from_t, edge_from + increase_idx), axis=0)
                # normalize weight to be in range (0, 1)
                weight = np.concatenate((weight_t, weight), axis=0)
                probs_t = weight_t / weight_t.max()
                probs = np.concatenate((probs_t, probs), axis=0)
                sigmas = np.concatenate((sigmas_t, sigmas), axis=0)
                rhos = np.concatenate((rhos_t, rhos), axis=0)
                feature_vectors = np.concatenate((fitting_data, feature_vectors), axis=0)
                attention = np.concatenate((attention_t, attention), axis=0)
                knn_indices = np.concatenate((knn_idxs_t, knn_indices+increase_idx), axis=0)
                # npr = np.concatenate((npr_t, npr), axis=0)
                time_step_nums.insert(0, (t_num, b_num))

        return edge_to, edge_from, weight, feature_vectors, time_step_nums, time_step_idxs_list, knn_indices, sigmas, rhos, attention
    

class SingleEpochSpatialEdgeConstructor(SpatialEdgeConstructor):
    def __init__(self, data_provider, iteration, s_n_epochs, b_n_epochs, n_neighbors, metric, sampler) -> None:
        super().__init__(data_provider, 100, s_n_epochs, b_n_epochs, n_neighbors, metric)
        self.iteration = iteration
        self.metric = metric
        self.sampler = sampler
    
    def construct(self):
        train_data = self.data_provider.train_representation(self.iteration)
        
        selected = self.sampler.sampling(train_data)
        train_data = train_data[selected]

        if self.b_n_epochs > 0:
            border_centers = self.data_provider.border_representation(self.iteration).squeeze()
            complex, sigmas, rhos, _ = self._construct_fuzzy_complex(train_data)
            bw_complex, _, _, _ = self._construct_boundary_wise_complex(train_data, border_centers, sigmas, rhos)
            edge_to, edge_from, weight = self._construct_step_edge_dataset(complex, bw_complex)
            feature_vectors = np.concatenate((train_data, border_centers), axis=0)
            # pred_model = self.data_provider.prediction_function(self.iteration)
            # attention = get_attention(pred_model, feature_vectors, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
            attention = np.zeros(feature_vectors.shape)
        elif self.b_n_epochs == 0:
            complex, _, _, _ = self._construct_fuzzy_complex(train_data)
            edge_to, edge_from, weight = self._construct_step_edge_dataset(complex, None)
            feature_vectors = np.copy(train_data)
            # pred_model = self.data_provider.prediction_function(self.iteration)
            # attention = get_attention(pred_model, feature_vectors, temperature=.01, device=self.data_provider.DEVICE, verbose=1)            
            attention = np.zeros(feature_vectors.shape)
        else: 
            raise Exception("Illegal border edges proposion!")
            
        return edge_to, edge_from, weight, feature_vectors, attention
    
    def record_time(self, save_dir, file_name, operation, t):
        file_path = os.path.join(save_dir, file_name+".json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                ti = json.load(f)
        else:
            ti = dict()
        if operation not in ti.keys():
            ti[operation] = dict()
        ti[operation][str(self.iteration)] = t
        with open(file_path, "w") as f:
            json.dump(ti, f)


class kcHybridSpatialEdgeConstructor(SpatialEdgeConstructor):
    def __init__(self, data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors, metric, MAX_HAUSDORFF, ALPHA, BETA, init_idxs=None, init_embeddings=None, c0=None, d0=None) -> None:
        super().__init__(data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors, metric)
        self.MAX_HAUSDORFF = MAX_HAUSDORFF
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.init_idxs = init_idxs
        self.init_embeddings = init_embeddings
        self.c0 = c0
        self.d0 = d0
    
    def _get_unit(self, data, adding_num=100):
        t0 = time.time()
        l = len(data)
        idxs = np.random.choice(np.arange(l), size=self.init_num, replace=False)

        id = IntrinsicDim(data)
        d0 = id.twonn_dimension_fast()

        kc = kCenterGreedy(data)
        _ = kc.select_batch_with_budgets(idxs, adding_num)
        c0 = kc.hausdorff()
        t1 = time.time()
        return c0, d0, "{:.1f}".format(t1-t0)
    
    def construct(self):
        """construct spatio-temporal complex and get edges

        Returns
        -------
        _type_
            _description_
        """

        # dummy input
        edge_to = None
        edge_from = None
        sigmas = None
        rhos = None
        weight = None
        probs = None
        feature_vectors = None
        attention = None
        knn_indices = None
        time_step_nums = list()
        time_step_idxs_list = list()
        coefficient = None
        embedded = None

        train_num = self.data_provider.train_num
        # load init_idxs
        if self.init_idxs is None:
            selected_idxs = np.random.choice(np.arange(train_num), size=self.init_num, replace=False)
        else:
            selected_idxs = np.copy(self.init_idxs)
        
        # load c0 d0
        if self.c0 is None or self.d0 is None:
            baseline_data = self.data_provider.train_representation(self.data_provider.e)
            max_x = np.linalg.norm(baseline_data, axis=1).max()
            baseline_data = baseline_data/max_x
            c0,d0,_ = self._get_unit(baseline_data)
            save_dir = os.path.join(self.data_provider.content_path, "selected_idxs")
            os.system("mkdir -p {}".format(save_dir))
            with open(os.path.join(save_dir,"baseline.json"), "w") as f:
                json.dump([float(c0), float(d0)], f)
            print("save c0 and d0 to disk!")
            
        else:
            c0 = self.c0
            d0 = self.d0

        # each time step
        for t in range(self.data_provider.e, self.data_provider.s - 1, -self.data_provider.p):
            print("=================+++={:d}=+++================".format(t))
            # load train data and border centers
            train_data = self.data_provider.train_representation(t).squeeze()

            # normalize data by max ||x||_2
            max_x = np.linalg.norm(train_data, axis=1).max()
            train_data = train_data/max_x

            # get normalization parameters for different epochs
            c,d,_ = self._get_unit(train_data)
            c_c0 = math.pow(c/c0, self.BETA)
            d_d0 = math.pow(d/d0, self.ALPHA)
            print("Finish calculating normaling factor")

            kc = kCenterGreedy(train_data)
            _, hausd = kc.select_batch_with_cn(selected_idxs, self.MAX_HAUSDORFF, c_c0, d_d0, p=0.95, return_min=True)
            selected_idxs = kc.already_selected.astype("int")

            save_dir = os.path.join(self.data_provider.content_path, "selected_idxs")
            os.system("mkdir -p {}".format(save_dir))
            with open(os.path.join(save_dir,"selected_{}.json".format(t)), "w") as f:
                json.dump(selected_idxs.tolist(), f)
            print("select {:d} points".format(len(selected_idxs)))

            time_step_idxs_list.insert(0, selected_idxs)

            train_data = self.data_provider.train_representation(t).squeeze()
            train_data = train_data[selected_idxs]

            if self.b_n_epochs != 0:
                # select highly used border centers...
                border_centers = self.data_provider.border_representation(t).squeeze()
                t_num = len(selected_idxs)
                b_num = len(border_centers)

                complex, sigmas_t1, rhos_t1, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                bw_complex, sigmas_t2, rhos_t2, _ = self._construct_boundary_wise_complex(train_data, border_centers)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, bw_complex)
                sigmas_t = np.concatenate((sigmas_t1, sigmas_t2[len(sigmas_t1):]), axis=0)
                rhos_t = np.concatenate((rhos_t1, rhos_t2[len(rhos_t1):]), axis=0)
                fitting_data = np.concatenate((train_data, border_centers), axis=0)
                pred_model = self.data_provider.prediction_function(t)
                attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
            else:
                t_num = len(selected_idxs)
                b_num = 0

                complex, sigmas_t, rhos_t, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, None)
                fitting_data = np.copy(train_data)
                pred_model = self.data_provider.prediction_function(t)
                attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
            
            if edge_to is None:
                edge_to = edge_to_t
                edge_from = edge_from_t
                weight = weight_t
                probs = weight_t / weight_t.max()
                feature_vectors = fitting_data
                attention = attention_t
                sigmas = sigmas_t
                rhos = rhos_t
                knn_indices = knn_idxs_t
                # npr = npr_t
                time_step_nums.insert(0, (t_num, b_num))

                if self.init_embeddings is None:
                    coefficient = np.zeros(len(feature_vectors))
                    embedded = np.zeros((len(feature_vectors), 2))
                else:
                    coefficient = np.zeros(len(feature_vectors))
                    coefficient[:len(self.init_embeddings)] = 1
                    embedded = np.zeros((len(feature_vectors), 2))
                    embedded[:len(self.init_embeddings)] = self.init_embeddings

            else:
                # every round, we need to add len(data) to edge_to(as well as edge_from) index
                increase_idx = len(fitting_data)
                edge_to = np.concatenate((edge_to_t, edge_to + increase_idx), axis=0)
                edge_from = np.concatenate((edge_from_t, edge_from + increase_idx), axis=0)
                # normalize weight to be in range (0, 1)
                weight = np.concatenate((weight_t, weight), axis=0)
                probs_t = weight_t / weight_t.max()
                probs = np.concatenate((probs_t, probs), axis=0)
                sigmas = np.concatenate((sigmas_t, sigmas), axis=0)
                rhos = np.concatenate((rhos_t, rhos), axis=0)
                feature_vectors = np.concatenate((fitting_data, feature_vectors), axis=0) 
                attention = np.concatenate((attention_t, attention), axis=0)
                knn_indices = np.concatenate((knn_idxs_t, knn_indices+increase_idx), axis=0)
                # npr = np.concatenate((npr_t, npr), axis=0)
                time_step_nums.insert(0, (t_num, b_num))
                coefficient = np.concatenate((np.zeros(len(fitting_data)), coefficient), axis=0)
                embedded = np.concatenate((np.zeros((len(fitting_data), 2)), embedded), axis=0)

        return edge_to, edge_from, weight, feature_vectors, embedded, coefficient, time_step_nums, time_step_idxs_list, knn_indices, sigmas, rhos, attention, (c0, d0)


class kcHybridDenseALSpatialEdgeConstructor(SpatialEdgeConstructor):
    def __init__(self, data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors, metric, MAX_HAUSDORFF, ALPHA, BETA, iteration, init_idxs=None, init_embeddings=None, c0=None, d0=None) -> None:
        super().__init__(data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors, metric)
        self.MAX_HAUSDORFF = MAX_HAUSDORFF
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.init_idxs = init_idxs
        self.init_embeddings = init_embeddings
        self.c0 = c0
        self.d0 = d0
        self.iteration = iteration
    
    def _get_unit(self, data, adding_num=100):
        t0 = time.time()
        l = len(data)
        idxs = np.random.choice(np.arange(l), size=self.init_num, replace=False)

        id = IntrinsicDim(data)
        d0 = id.twonn_dimension_fast()

        kc = kCenterGreedy(data)
        _ = kc.select_batch_with_budgets(idxs, adding_num)
        c0 = kc.hausdorff()
        t1 = time.time()
        return c0, d0, "{:.1f}".format(t1-t0)
    
    def construct(self):
        """construct spatio-temporal complex and get edges

        Returns
        -------
        _type_
            _description_
        """

        # dummy input
        edge_to = None
        edge_from = None
        sigmas = None
        rhos = None
        weight = None
        probs = None
        feature_vectors = None
        attention = None
        knn_indices = None
        time_step_nums = list()
        time_step_idxs_list = list()
        coefficient = None
        embedded = None

        train_num = self.data_provider.label_num(self.iteration)
        # load init_idxs
        if self.init_idxs is None:
            selected_idxs = np.random.choice(np.arange(train_num), size=self.init_num, replace=False)
        else:
            selected_idxs = np.copy(self.init_idxs)
        
        # load c0 d0
        if self.c0 is None or self.d0 is None:
            baseline_data = self.data_provider.train_representation_lb(self.iteration, self.data_provider.e)
            max_x = np.linalg.norm(baseline_data, axis=1).max()
            baseline_data = baseline_data/max_x
            c0,d0,_ = self._get_unit(baseline_data)
            save_dir = os.path.join(self.data_provider.content_path, "Model", "Iteration_{}".format(self.iteration), "selected_idxs")
            os.system("mkdir -p {}".format(save_dir))
            with open(os.path.join(save_dir,"baseline.json"), "w") as f:
                json.dump([float(c0), float(d0)], f)
            print("save c0 and d0 to disk!")
            
        else:
            c0 = self.c0
            d0 = self.d0

        # each time step
        for t in range(self.data_provider.e, self.data_provider.s - 1, -self.data_provider.p):
            print("=================+++={:d}=+++================".format(t))
            # load train data and border centers
            train_data = self.data_provider.train_representation_lb(self.iteration, t).squeeze()

            # normalize data by max ||x||_2
            max_x = np.linalg.norm(train_data, axis=1).max()
            train_data = train_data/max_x

            # get normalization parameters for different epochs
            c,d,_ = self._get_unit(train_data)
            c_c0 = math.pow(c/c0, self.BETA)
            d_d0 = math.pow(d/d0, self.ALPHA)
            print("Finish calculating normaling factor")

            kc = kCenterGreedy(train_data)
            _, hausd = kc.select_batch_with_cn(selected_idxs, self.MAX_HAUSDORFF, c_c0, d_d0, p=0.95, return_min=True)
            selected_idxs = kc.already_selected.astype("int")

            save_dir = os.path.join(self.data_provider.content_path, "Model", "Iteration_{}".format(self.iteration), "selected_idxs")
            os.system("mkdir -p {}".format(save_dir))
            with open(os.path.join(save_dir,"selected_{}.json".format(t)), "w") as f:
                json.dump(selected_idxs.tolist(), f)
            print("select {:d} points".format(len(selected_idxs)))

            time_step_idxs_list.insert(0, selected_idxs)

            train_data = self.data_provider.train_representation_lb(self.iteration, t).squeeze()
            train_data = train_data[selected_idxs]

            if self.b_n_epochs != 0:
                # select highly used border centers...
                border_centers = self.data_provider.border_representation(self.iteration, t).squeeze()
                t_num = len(selected_idxs)
                b_num = len(border_centers)

                complex, sigmas_t1, rhos_t1, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                bw_complex, sigmas_t2, rhos_t2, _ = self._construct_boundary_wise_complex(train_data, border_centers)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, bw_complex)
                sigmas_t = np.concatenate((sigmas_t1, sigmas_t2[len(sigmas_t1):]), axis=0)
                rhos_t = np.concatenate((rhos_t1, rhos_t2[len(rhos_t1):]), axis=0)
                fitting_data = np.concatenate((train_data, border_centers), axis=0)
                pred_model = self.data_provider.prediction_function(t)
                attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
            else:
                t_num = len(selected_idxs)
                b_num = 0

                complex, sigmas_t, rhos_t, knn_idxs_t = self._construct_fuzzy_complex(train_data)
                edge_to_t, edge_from_t, weight_t = self._construct_step_edge_dataset(complex, None)
                fitting_data = np.copy(train_data)
                pred_model = self.data_provider.prediction_function(self.iteration,t)
                attention_t = get_attention(pred_model, fitting_data, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
            
            if edge_to is None:
                edge_to = edge_to_t
                edge_from = edge_from_t
                weight = weight_t
                probs = weight_t / weight_t.max()
                feature_vectors = fitting_data
                attention = attention_t
                sigmas = sigmas_t
                rhos = rhos_t
                knn_indices = knn_idxs_t
                # npr = npr_t
                time_step_nums.insert(0, (t_num, b_num))

                if self.init_embeddings is None:
                    coefficient = np.zeros(len(feature_vectors))
                    embedded = np.zeros((len(feature_vectors), 2))
                else:
                    coefficient = np.zeros(len(feature_vectors))
                    coefficient[:len(self.init_embeddings)] = 1
                    embedded = np.zeros((len(feature_vectors), 2))
                    embedded[:len(self.init_embeddings)] = self.init_embeddings

            else:
                # every round, we need to add len(data) to edge_to(as well as edge_from) index
                increase_idx = len(fitting_data)
                edge_to = np.concatenate((edge_to_t, edge_to + increase_idx), axis=0)
                edge_from = np.concatenate((edge_from_t, edge_from + increase_idx), axis=0)
                # normalize weight to be in range (0, 1)
                weight = np.concatenate((weight_t, weight), axis=0)
                probs_t = weight_t / weight_t.max()
                probs = np.concatenate((probs_t, probs), axis=0)
                sigmas = np.concatenate((sigmas_t, sigmas), axis=0)
                rhos = np.concatenate((rhos_t, rhos), axis=0)
                feature_vectors = np.concatenate((fitting_data, feature_vectors), axis=0) 
                attention = np.concatenate((attention_t, attention), axis=0)
                knn_indices = np.concatenate((knn_idxs_t, knn_indices+increase_idx), axis=0)
                # npr = np.concatenate((npr_t, npr), axis=0)
                time_step_nums.insert(0, (t_num, b_num))
                coefficient = np.concatenate((np.zeros(len(fitting_data)), coefficient), axis=0)
                embedded = np.concatenate((np.zeros((len(fitting_data), 2)), embedded), axis=0)

        return edge_to, edge_from, weight, feature_vectors, embedded, coefficient, time_step_nums, time_step_idxs_list, knn_indices, sigmas, rhos, attention, (c0, d0)


class tfEdgeConstructor(SpatialEdgeConstructor):
    def __init__(self, data_provider, s_n_epochs, b_n_epochs, n_neighbors, metric) -> None:
        super().__init__(data_provider, 100, s_n_epochs, b_n_epochs, n_neighbors, metric)
    # override
    def _construct_step_edge_dataset(self, vr_complex, bw_complex):
        """
        construct the mixed edge dataset for one time step
            connect border points and train data(both direction)
        :param vr_complex: Vietoris-Rips complex
        :param bw_complex: boundary-augmented complex
        :param n_epochs: the number of epoch that we iterate each round
        :return: edge dataset
        """
        # get data from graph
        _, vr_head, vr_tail, vr_weight, _ = get_graph_elements(vr_complex, self.s_n_epochs)
        epochs_per_sample = make_epochs_per_sample(vr_weight, 10)
        vr_head = np.repeat(vr_head, epochs_per_sample.astype("int"))
        vr_tail = np.repeat(vr_tail, epochs_per_sample.astype("int"))
        vr_weight = np.repeat(vr_weight, epochs_per_sample.astype("int"))
        
        # get data from graph
        if self.b_n_epochs == 0:
            return vr_head, vr_tail, vr_weight
        else:
            _, bw_head, bw_tail, bw_weight, _ = get_graph_elements(bw_complex, self.b_n_epochs)
            b_epochs_per_sample = make_epochs_per_sample(bw_weight, self.b_n_epochs)
            bw_head = np.repeat(bw_head, b_epochs_per_sample.astype("int"))
            bw_tail = np.repeat(bw_tail, b_epochs_per_sample.astype("int"))
            bw_weight = np.repeat(bw_weight, epochs_per_sample.astype("int"))
            head = np.concatenate((vr_head, bw_head), axis=0)
            tail = np.concatenate((vr_tail, bw_tail), axis=0)
            weight = np.concatenate((vr_weight, bw_weight), axis=0)
        return head, tail, weight
        
    def construct(self, prev_iteration, iteration):
        '''
        If prev_iteration<epoch_start, then temporal loss will be 0
        '''
        train_data = self.data_provider.train_representation(iteration)
        if prev_iteration > self.data_provider.s:
            prev_data = self.data_provider.train_representation(prev_iteration)
        else:
            prev_data = None
        n_rate = find_neighbor_preserving_rate(prev_data, train_data, self.n_neighbors)
        if self.b_n_epochs > 0:
            border_centers = self.data_provider.border_representation(iteration).squeeze()
            complex, _, _, _ = self._construct_fuzzy_complex(train_data)
            bw_complex, _, _, _ = self._construct_boundary_wise_complex(train_data, border_centers)
            edges_to_exp, edges_from_exp, weights_exp = self._construct_step_edge_dataset(complex, bw_complex)
            feature_vectors = np.concatenate((train_data, border_centers), axis=0)
            # pred_model = self.data_provider.prediction_function(self.iteration)
            # attention = get_attention(pred_model, feature_vectors, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
            attention = np.zeros(feature_vectors.shape)

        elif self.b_n_epochs == 0:
            complex, _, _, _ = self._construct_fuzzy_complex(train_data)
            edges_to_exp, edges_from_exp, weights_exp = self._construct_step_edge_dataset(complex, None)
            feature_vectors = np.copy(train_data)
            # pred_model = self.data_provider.prediction_function(self.iteration)
            # attention = get_attention(pred_model, feature_vectors, temperature=.01, device=self.data_provider.DEVICE, verbose=1)            
            attention = np.zeros(feature_vectors.shape)
        else: 
            raise Exception("Illegal border edges proposion!")
            
        return edges_to_exp, edges_from_exp, weights_exp, feature_vectors, attention, n_rate


class SplitSpatialTemporalEdgeConstructor(SpatialEdgeConstructor):
    '''split temporal and spatial edges, then train them separately'''
    def __init__(self, data_provider, projector, s_n_epochs, b_n_epochs, t_n_epochs, n_neighbors, metric, sampler) -> None:
        super().__init__(data_provider, 100, s_n_epochs, b_n_epochs, n_neighbors, metric)
        self.t_n_epochs = t_n_epochs
        self.projector = projector
        self.sampler = sampler
    '''
    3 strategies:

    1. sequence
    2. skip some epoch
    3. estimate by nearest visualizer
    '''
    # TODO: move this class to temporal edge construction
    def _uncertainty_measure(self, prev_e, next_e):
        prev_data = self.data_provider.train_representation(prev_e)
        next_data = self.data_provider.train_representation(next_e)

        mean_t = prev_data.mean(axis=0)
        std_t = prev_data.std(axis=0)
        mean_tk = next_data.mean(axis=0)
        std_tk = next_data.std(axis=0)

        z_score = np.abs(mean_tk - mean_t) / (np.sqrt(std_t ** 2 + std_tk ** 2)+5E-4)
        margin = np.mean(z_score)
        return margin

    def _construct_single_memory(self, curr, prev):
        # load train data and border centers
        prev_data = self.data_provider.train_representation(prev)
        # count dist
        dist = self._uncertainty_measure(prev, curr)

        # TODO how much prev data used
        selected = np.random.choice(len(prev_data), int(0.05*len(prev_data)), replace=False)
        prev_data = prev_data[selected]

        # use prediction as estimation
        prev_embedded = self.projector.batch_project(prev, prev_data)
        # xy_range = np.ptp(prev_embedded, axis=0)

        # TODO clip margin?
        # dist = np.clip(dist, a_min=0, a_max=0.1)
        margin = dist * np.ones(len(prev_embedded))
        margin = np.clip(margin, a_min=0.1, a_max=1.0)
        return prev_data, prev_embedded, margin
    
    def _construct_single_memory_estimation(self, curr, prev, pseudo_epoch):
        '''use the fake_prev to estimate prev'''

        prev_data = self.data_provider.train_representation(prev)
        dist = self._uncertainty_measure(prev, curr)

        # TODO how much prev data used
        selected = np.random.choice(len(prev_data), int(0.05*len(prev_data)), replace=False)
        prev_data = prev_data[selected]

        # use prediction as estimation
        prev_embedded = self.projector.batch_project(pseudo_epoch, prev_data)
        # xy_range = np.ptp(prev_embedded, axis=0)

        # TODO clip margin?
        # dist = np.clip(dist, a_min=0, a_max=0.1)
        # set to be two times bigger than sequence
        margin = dist* np.ones(len(prev_embedded))
        margin = 2*np.clip(margin, a_min=0.1, a_max=1.0)
        return prev_data, prev_embedded, margin

    def _construct_working_memory(self, curr):
        # construct working history
        working_epochs = list()
        i = 0
        while True:
            prev = curr - pow(2, i) * self.data_provider.p
            i = i+1
            if prev >= self.data_provider.s:
                working_epochs.append(prev)
            else:
                break

        # based epoch
        based_epoch = curr - self.data_provider.p
        # working momery
        read_memory_data = None
        read_memory_embedded = None
        margins = None
        for i in working_epochs:
            prev_data, prev_embedded, margin_tmp = self._construct_single_memory(curr, i)
            if read_memory_data is None:
                read_memory_data = np.copy(prev_data)
                read_memory_embedded = np.copy(prev_embedded)
                margins = np.copy(margin_tmp)
            else:
                read_memory_data = np.concatenate((read_memory_data, prev_data), axis=0)
                read_memory_embedded = np.concatenate((read_memory_embedded, prev_embedded), axis=0)
                margins = np.concatenate((margins, margin_tmp), axis=0)
        return read_memory_data, read_memory_embedded, margins, based_epoch
    
    def _construct_working_memory_skip(self, curr, resume):
        # # construct working history
        # working_epochs = list()
        # i = 0
        # # if less than missing window, dont skip
        # if curr - (missing_window+1) * self.data_provider.p < self.data_provider.s:
        #     skip = 0
        # else:
        #     skip = missing_window
        # based_epoch = curr - self.data_provider.p * (skip+1)
        working_epochs = [resume]
        i = 0
        while True:
            prev = resume - pow(2, i) * self.data_provider.p
            i = i+1
            if prev >= self.data_provider.s:
                working_epochs.append(prev)
            else:
                break
        # working momery
        read_memory_data = None
        read_memory_embedded = None
        margins = None
        for i in working_epochs:
            prev_data, prev_embedded, margin_tmp = self._construct_single_memory(curr, i)
            if read_memory_data is None:
                read_memory_data = np.copy(prev_data)
                read_memory_embedded = np.copy(prev_embedded)
                margins = np.copy(margin_tmp)
            else:
                read_memory_data = np.concatenate((read_memory_data, prev_data), axis=0)
                read_memory_embedded = np.concatenate((read_memory_embedded, prev_embedded), axis=0)
                margins = np.concatenate((margins, margin_tmp), axis=0)
        return read_memory_data, read_memory_embedded, margins

    def _construct_working_memory_estimation(self, curr, missing_window):
        # construct working history
        working_epochs = list()
        i = 0
        while True:
            prev = curr - pow(2, i) * self.data_provider.p
            i = i+1
            if prev >= self.data_provider.s:
                working_epochs.append(prev)
            else:
                break
        # working momery
        read_memory_data = None
        read_memory_embedded = None
        margins = None

        for i in working_epochs:
            # use prediction as estimation
            if i <= self.data_provider.s + missing_window*self.data_provider.p:
                prev_data, prev_embedded, margin_tmp = self._construct_single_memory(curr, i)
            elif i > (curr- (missing_window+1)*self.data_provider.p):
                curr_available_epoch = self.data_provider.s + (missing_window-1) * self.data_provider.p
                pseudo_epoch = curr- (missing_window+1)*self.data_provider.p
                # use init ones as pseudo epoch
                # pseudo_epoch = i - (missing_window+1)*self.data_provider.p
                pseudo_epoch = max(pseudo_epoch, curr_available_epoch)
                prev_data, prev_embedded, margin_tmp = self._construct_single_memory_estimation(curr, i, pseudo_epoch)
            else:
                prev_data, prev_embedded, margin_tmp = self._construct_single_memory(curr, i)

        # based_epoch
        based_epoch = working_epochs[0]
        if based_epoch <= self.data_provider.s + missing_window*self.data_provider.p:
            based_epoch = curr - self.data_provider.p
        elif based_epoch > (curr- (missing_window+1)*self.data_provider.p):
            curr_available_epoch = self.data_provider.s + (missing_window-1) * self.data_provider.p
            pseudo_epoch = curr- (missing_window+1)*self.data_provider.p
            based_epoch = max(pseudo_epoch, curr_available_epoch)
        else:
            based_epoch = curr - self.data_provider.p

            if read_memory_data is None:
                read_memory_data = np.copy(prev_data)
                read_memory_embedded = np.copy(prev_embedded)
                margins = np.copy(margin_tmp)
            else:
                read_memory_data = np.concatenate((read_memory_data, prev_data), axis=0)
                read_memory_embedded = np.concatenate((read_memory_embedded, prev_embedded), axis=0)
                margins = np.concatenate((margins, margin_tmp), axis=0)
        return read_memory_data, read_memory_embedded, margins, based_epoch
        
    def construct(self, next_iter, resume):
        prev_data, prev_embedded, margins = self._construct_working_memory_skip(next_iter, resume)
        
        next_data = self.data_provider.train_representation(next_iter)
        selected = self.sampler.sampling(next_data)
        next_data = next_data[selected]

        # temporal edges
        t_complex = self._construct_local_temporal_complex(prev_data, next_data, b_num=0)
        edge_t_to, edge_t_from, weight_t = self._construct_step_edge_dataset(None, None, t_complex)

        if self.b_n_epochs > 0:
            border_centers = self.data_provider.border_representation(next_iter).squeeze()
            complex, sigmas, rhos, _ = self._construct_fuzzy_complex(next_data)
            bw_complex, _, _, _ = self._construct_boundary_wise_complex(next_data, border_centers, sigmas, rhos)
            edge_to, edge_from, weight = self._construct_step_edge_dataset(complex, bw_complex, None)
            feature_vectors = np.concatenate((next_data, border_centers), axis=0)
            # pred_model = self.data_provider.prediction_function(self.iteration)
            # attention = get_attention(pred_model, feature_vectors, temperature=.01, device=self.data_provider.DEVICE, verbose=1)
            attention = np.zeros(feature_vectors.shape)
        elif self.b_n_epochs == 0:
            complex, _, _, _ = self._construct_fuzzy_complex(next_data)
            edge_to, edge_from, weight = self._construct_step_edge_dataset(complex, None, None)
            feature_vectors = np.copy(next_data)
            # pred_model = self.data_provider.prediction_function(self.iteration)
            # attention = get_attention(pred_model, feature_vectors, temperature=.01, device=self.data_provider.DEVICE, verbose=1)            
            attention = np.zeros(feature_vectors.shape)
        else: 
            raise Exception("Illegal border edges proposion!")
        
        # spatial component, temporal component
        return (edge_to, edge_from, weight, feature_vectors, attention), (edge_t_to, edge_t_from, weight_t, next_data, prev_data, prev_embedded, margins)
    
    def record_time(self, save_dir, file_name, operation, t):
        file_path = os.path.join(save_dir, file_name+".json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                ti = json.load(f)
        else:
            ti = dict()
        if operation not in ti.keys():
            ti[operation] = dict()
        ti[operation][str(self.iteration)] = t
        with open(file_path, "w") as f:
            json.dump(ti, f)