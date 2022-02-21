import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import Normal
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from treeflow.traversal.anchor_heights import get_anchor_heights_tensor
from treeflow.bijectors.node_height_ratio_bijector import NodeHeightRatioChainBijector
from treeflow.tree.io import parse_newick
import treeflow_pipeline.results


def simulate_heights(config, topology_file, seed, newick_out):
    topology_tree = convert_tree_to_tensor(parse_newick(topology_file))
    anchor_heights = get_anchor_heights_tensor(
        topology_tree.topology, topology_tree.sampling_times
    )
    height_bijector = NodeHeightRatioChainBijector(
        topology_tree.topology, anchor_heights
    )
    loc = height_bijector.inverse(topology_tree.node_heights)
    base_dist = Normal(loc=loc, scale=config["height_scale"])
    height_samples = height_bijector.forward(
        base_dist.sample(config["sample_count"], seed=seed)
    )
    samples = topology_tree.with_node_heights(height_samples)
    treeflow_pipeline.results.write_tensor_trees(
        topology_file, samples.branch_lengths.numpy(), newick_out
    )
    return samples
