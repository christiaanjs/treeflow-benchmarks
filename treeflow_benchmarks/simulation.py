import numpy as np
from tensorflow_probability.python.distributions import Normal
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    convert_tree_to_tensor,
)
from treeflow.traversal.anchor_heights import get_anchor_heights_tensor
from treeflow.bijectors.node_height_ratio_bijector import NodeHeightRatioChainBijector
from treeflow.tree.io import parse_newick
import treeflow_pipeline.results
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree


def simulate_heights(config, topology_file, seed, newick_out) -> NumpyRootedTree:
    numpy_topology_tree = parse_newick(topology_file)
    topology_tree = convert_tree_to_tensor(numpy_topology_tree)
    anchor_heights = get_anchor_heights_tensor(
        topology_tree.topology, topology_tree.sampling_times
    )
    height_bijector = NodeHeightRatioChainBijector(
        topology_tree.topology, anchor_heights
    )
    loc = height_bijector.inverse(topology_tree.node_heights)
    base_dist = Normal(loc=loc, scale=config["height_scale"])
    sample_size = config["sample_count"]
    height_samples = height_bijector.forward(base_dist.sample(sample_size, seed=seed))
    sampling_times = np.broadcast_to(
        numpy_topology_tree.sampling_times,
        (sample_size, numpy_topology_tree.taxon_count),
    )
    samples = NumpyRootedTree(
        node_heights=height_samples.numpy(),
        sampling_times=sampling_times,
        topology=numpy_topology_tree.topology,
    )
    treeflow_pipeline.results.write_tensor_trees(
        topology_file, samples.branch_lengths, newick_out
    )
    return samples
