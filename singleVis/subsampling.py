'''
One time subsampling for efficiency
Random Sampling

Coreset Sampling

Density-aware subsampling method for downsampling data.
1. select only sparse region
2. density-aware hierachy sampling
3. considering the uncertainty
papers:
- In Defense of Core-set: A Density-aware Core-set Selection for Active Learning
- real: a representative error-driven approach for active learning
'''
from abc import ABC, abstractmethod

from sklearn.neighbors import NearestNeighbors

# from jenkspy import JenksNaturalBreaks
import numpy as np
from scipy.special import softmax

from singleVis.kcenter_greedy import kCenterGreedy
from singleVis.utils import knn

class SubSampling(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def sampling(self, *args, **kwargs):
        pass


class IdentitySampling(SubSampling):
    def __init__(self, verbose=1) -> None:
        super().__init__()
        self.name = "Identity"
        self.verbose = verbose
    
    def sampling(self, data):
        if self.verbose:
            print("Sampling 100% data points")
        return np.arange(len(data))


class RandomSampling(SubSampling):
    def __init__(self, ratio, verbose=1) -> None:
        super().__init__()
        self.ratio = ratio
        self.name = "Random"
        self.verbose = verbose
    
    def sampling(self, data):
        num = len(data)
        selected_idxs = np.random.choice(num, int(self.ratio*num), replace=False)
        if self.verbose:
            print(f"Sampling {self.ratio*100}% data points")
        return selected_idxs
    

class DensityAwareSampling(SubSampling):
    '''Dynamically choose a ratio such that the changing density is minimized'''
    def __init__(self, verbose=1) -> None:
        super().__init__()
        self.name = "DensityAware"
        self.verbose = verbose
    
    def density_estimation(self, data, estimated_ratio=0.5, repeat=2, k=20, metric="euclidean"):
        avg_dists = np.zeros(repeat)
        for r in range(repeat):
            sampling_idxs = np.random.choice(len(data), size=int(len(data)*estimated_ratio), replace=False)
            high_neigh = NearestNeighbors(n_neighbors=k, metric=metric)
            high_neigh.fit(data)
            knn_dists, _ = high_neigh.kneighbors(data[sampling_idxs], n_neighbors=k, return_distance=True)
            avg_dists[r] = knn_dists[:, -1].mean()
        return avg_dists.mean()
    
    def sampling(self, data, threshold=0.25):
        # TODO binery seach
        avg_dist = self.density_estimation(data, estimated_ratio=0.01, repeat=5)
        ratios = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        densities = list()
        for ratio in ratios:
            subsampling_idxs = np.random.choice(len(data), int(ratio*len(data)), replace=False)
            estimated_avg_distance = self.density_estimation(data[subsampling_idxs], estimated_ratio=0.25)
            densities.append(avg_dist/estimated_avg_distance)
        densities = np.array(densities)
        dx = (densities-1)/(ratios-1)
        target_ratio = ratios[-1]
        for i in range(len(dx)):
            if dx[i]<=threshold:
                target_ratio = ratios[i]
                if self.verbose:
                    print(f"Sampling {target_ratio*100}% data points")
                return np.random.choice(len(data), int(target_ratio*len(data)), replace=False)
        if self.verbose:
            print(f"Sampling {target_ratio*100}% data points")
        return np.random.choice(len(data), int(target_ratio*len(data)), replace=False)
        

class CoresetSampling(SubSampling):
    def __init__(self, ratio, metric, verbose=1) -> None:
        super().__init__()
        self.ratio = ratio
        self.metric = metric
        self.name = "Coreset"
        self.verbose = verbose

    def sampling(self, data):
        target_num = int(len(data)*self.ratio)
        assert target_num > 100
        random_init_idxs = np.random.choice(len(data), size=100, replace=False)
        kc = kCenterGreedy(data, metric=self.metric)
        _ = kc.select_batch_with_budgets(random_init_idxs, target_num-100)
        if verbose:
            print(f"Sampling {len(kc.already_selected)/len(data)*100}% data points")
        return kc.already_selected


# class DensityAwareSampling(SubSampling):

#     def __init__(self, n_classes, n_neighbors, ratio, temperature, metric) -> None:
#         super().__init__()
#         self.n_classes = n_classes
#         self.n_neighbors = n_neighbors
#         self.metric = metric
#         self.ratio = ratio
#         self.temperature = temperature
#         self._compute_density(k)

#     def _compute_density(self, data):
#         _, knn_dists = knn(data, self.n_neighbors, self.metric)
#         avg_distance = knn_dists[:, -1]
#         density = self.n_neighbors / avg_distance
#         # assign density as a property
#         self.density = density
#         print("Finish computing density...")
#         return density, avg_distance.mean()

#     def sampling(self, data):
#         num = len(data)
#         target_num = int(len(data)*self.ratio)

#         # separate data into n_classes groups
#         print("Start to categorize density...")
#         jnb = JenksNaturalBreaks(self.n_classes)
#         jnb.fit(self.density)
#         print("Finish to categorize density...")
#         # [0 0 0 1 1 1 2 2 2 3 3 3]
#         self.labels = jnb.labels_

#         # assign a selection ratio to each group
#         self.group = list()
#         group_nums = list()
#         for group in range(self.n_classes):
#             group_idxs = np.argwhere(self.labels == group).squeeze()
#             self.group.append(group_idxs)
#             group_nums.append(len(group_idxs))

#         group_nums = np.array(group_nums)
#         group_ratios = softmax((1 - group_nums/num)/self.temperature)
#         # group_ratios = softmax((group_nums/num)/temperature)
#         group_selected_num = (target_num*group_ratios).astype(int)
#         print(f"density: {self.density}\nSelected num: {group_selected_num}\nGroup num:{group_nums}\n")

#         # coreset sampling for each region
#         selected_idxs = list()
#         for group, selected_num in zip(self.group, group_selected_num):
#             # in this case we use random
#             # in case not enough group num
#             if selected_num>= len(group):
#                 selected_idxs.extend(group.tolist())
#             else:
#                 assert selected_num > 100
#                 kc = kCenterGreedy(data[group], metric=self.metric)
#                 random_init_idxs = np.random.choice(len(group), size=100, replace=False)
#                 _ = kc.select_batch_with_budgets(random_init_idxs, selected_num-100)
#                 selected_group_idxs = group[kc.already_selected]
                
#                 # selected_group_idxs = np.random.choice(group, selected_num, replace=False)
#                 selected_idxs.extend(selected_group_idxs.tolist())
            
#         # return selected_idxs
#         print(f"Select {len(selected_idxs)} data points...")
#         return np.asarray(selected_idxs)

# random sampling
# train_data = data_provider.train_representation(I)

# selected = np.random.choice(len(train_data), int(RATIO*len(train_data)), replace=False)
# train_data = train_data[selected]

# kc = kCenterGreedy(train_data)
# selected_idxs = np.random.choice(len(train_data), 200, replace=False)
# kc.select_batch_with_budgets(selected_idxs, budgets=int(ratio*len(train_data))-200)
# selected_idxs = kc.already_selected.astype("int")
# train_data = train_data[selected_idxs]

# farthest point sampling
# from dgl.geometry import farthest_point_sampler
# data = torch.from_numpy(train_data[np.newaxis,:,:]).to(device=torch.device("cuda:1"))
# point_idxs = farthest_point_sampler(data, int(ratio*len(train_data)))
# point_idxs = point_idxs.cpu().numpy().squeeze(0)
# train_data = train_data[point_idxs]

# decision set, sampling samples with lower confidence
# preds = data_provider.get_pred(I, train_data)
# from scipy.special import softmax
# probs = 1 - softmax(preds, axis=1).max(axis=1)
# probs_ = probs/probs.sum()
# selected_1 = np.random.choice(len(train_data), int(0.9*ratio*len(train_data)), replace=False, p=probs_)
# train_data_ = train_data[selected_1]
# probs_ = (1-probs)/(1-probs).sum()
# selected_2 = np.random.choice(len(train_data), int(0.1*ratio*len(train_data)), replace=False, p=probs_)
# train_data = np.concatenate((train_data_, train_data[selected_2]))

# # density based sampling
# _, density = density_estimation(train_data)
# selected = np.argsort(density)[:int(RATIO*len(train_data))]
# train_data = train_data[selected]



        