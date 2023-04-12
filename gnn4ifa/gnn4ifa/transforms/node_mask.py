from typing import Optional, Tuple, Union
import random
import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
# Import modules
from gnn4ifa.utils import timeit


class RandomNodeMasking(BaseTransform):
    r"""Performs a random mask over nodes of the graph given in input.
    Args:
        sampling_strategy (str, optional): Sampling strategy to leverage to identify nodes to be masked.
                    'local' strategy iterates over each node of the graph and mask each node
                    with probability p=sampling_probability.
                    'global' strategy picks randomly p*n_nodes indices from the list of nodes.
                    Where p is the sampling probability and n_nodes is the number of nodes in the graph.
        sampling_probability (float, optional): Sampling probability with which each node is masked.
    """

    def __init__(
            self,
            sampling_strategy: Optional[str] = 'local',
            sampling_probability: Optional[float] = .3,
    ):
        assert sampling_strategy in ['global', 'local']
        self.sampling_strategy: str = sampling_strategy
        self.sampling_probability: float = sampling_probability
        self.dummy_feature_vector: Tensor = None

    def update_feature_vector(self,
                              dummy_feature_vector: Tensor) -> None:
        self.dummy_feature_vector: Tensor = dummy_feature_vector

    #@timeit
    def __call__(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        # Get data from the sample received
        node_features: Tensor = data.x.detach().clone()
        # Store original node features in the data
        data.original_x: Tensor = data.x.detach().clone()
        # Sample randomly the nodes depending on the sampling strategy defined
        if self.sampling_strategy == 'global':
            # Randomly permutes the node features
            permutation: Tensor = torch.randperm(node_features.shape[0])
            # Define the number of nodes to mask depending on the sampling probability
            n_samples: int = int(data.num_nodes * self.sampling_probability)
            # Get indices of sampled edges
            sample_indices: Tensor = torch.unsqueeze(permutation[:n_samples], dim=0)
        elif self.sampling_strategy == 'local':
            # Define while to avoid the situation where no nodes are sampled
            at_least_one_sample = False
            while not at_least_one_sample:
                # Define empty tensor of sampled indices
                sample_indices: List[int] = []
                # Iterate over each node in the graph
                for node_i in range(node_features.shape[0]):
                    # Sample the node with probability sampling_probability
                    if random.random() < 0.3:
                        sample_indices.append(node_i)
                    else:
                        continue
                if len(sample_indices) > 0:
                    at_least_one_sample = True
            # Convert list to tensor
            sample_indices: Tensor = torch.unsqueeze(torch.tensor(sample_indices, dtype=torch.int64), dim=0)
        else:
            raise ValueError('Sampling strategy {} not yet supported'.format(self.sampling_strategy))
        # Define empty dummy feature vector if it was never initialized
        if self.dummy_feature_vector is None:
            self.dummy_feature_vector: Tensor = torch.zeros((node_features.shape[1],), dtype=torch.float64)
        # Mask the nodes using the dummy feature vector at hand
        node_features[sample_indices, :] = self.dummy_feature_vector
        # Set back data.x to be node_features
        data.x = node_features.detach().clone()
        # Append to data the list of sample_indices
        if sample_indices.shape[1] == 1:
            data.masked_nodes_indices = sample_indices.detach().clone().squeeze().unsqueeze(dim=0)
        else:
            data.masked_nodes_indices = sample_indices.detach().clone().squeeze()
        return data

    def __repr__(self) -> str:
        return '{}(sampl={}_prob={})'.format(self.__class__.__name__,
                                             self.sampling_strategy,
                                             self.sampling_probability)
