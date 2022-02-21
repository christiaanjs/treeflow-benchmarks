import typing as tp
import tensorflow as tf
import treeflow_benchmarks.benchmarking
from treeflow.bijectors.node_height_ratio_bijector import NodeHeightRatioBijector
from treeflow.tree.io import parse_newick
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    convert_tree_to_tensor,
    TensorflowRootedTree,
)
from treeflow.traversal.anchor_heights import get_anchor_heights_tensor


def get_bij(
    topology_file, _class=NodeHeightRatioBijector, **kwargs
) -> tp.Tuple[NodeHeightRatioBijector, TensorflowRootedTree]:
    tree = convert_tree_to_tensor(parse_newick(topology_file))
    anchor_heights = get_anchor_heights_tensor(tree.topology, tree.sampling_times)
    bij = _class(topology=tree.topology, anchor_heights=anchor_heights, **kwargs)
    return bij, tree


def get_ratios(topology_file, trees: TensorflowRootedTree):
    bij, tree = get_bij(topology_file)
    return tf.function(bij.inverse)(trees.node_heights)


class TensorflowRatioTransformBenchmarkable(
    treeflow_benchmarks.benchmarking.RatioTransformBenchmarkable
):
    def initialize(self, topology_file):
        self.bij, self.tree = get_bij(topology_file)
        self.forward = tf.function(self.bij.forward)

        def grad(ratios, height_gradients):
            with tf.GradientTape() as t:
                t.watch(ratios)
                heights = self.bij.forward(ratios)
                return t.gradient(heights, ratios, output_gradients=height_gradients)

        self.grad = tf.function(grad)

        ratios = self.bij.inverse(self.tree.node_heights)
        self.forward(ratios)  # Do tracing
        self.grad(ratios, tf.ones_like(ratios))

    def calculate_heights(self, ratios):
        return self.forward(ratios)

    def calculate_ratio_gradients(self, ratios, heights, height_gradients):
        return self.grad(ratios, height_gradients)
