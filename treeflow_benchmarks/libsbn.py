import tensorflow as tf
import numpy as np

from treeflow_benchmarks.benchmarking import (
    LikelihoodBenchmarkable,
    RatioTransformBenchmarkable,
)
from treeflow_benchmarks.tree_transform import get_bij
from treeflow_pipeline.model import get_likelihood
import treeflow.tree_processing
import treeflow.tree_transform
import treeflow.libsbn
import treeflow
import libsbn


class BeagleLikelihoodBenchmarkable(LikelihoodBenchmarkable):
    def initialize(self, topology_file, fasta_file, model):
        starting_values = dict(model.subst_params)
        for key in list(model.subst_params.keys()):
            model.subst_params[key] = "fixed"
        log_prob, self.inst = get_likelihood(
            topology_file, fasta_file, starting_values, model, dict(rescaling=False)
        )

        def grad(branch_lengths):
            with tf.GradientTape() as t:
                t.watch(branch_lengths)
                val = self.log_prob(branch_lengths)
                return t.gradient(val, branch_lengths)

        self.log_prob = tf.function(log_prob)
        self.grad = tf.function(grad)

        branch_lengths = np.array(self.inst.tree_collection.trees[0].branch_lengths)[
            :-1
        ]
        self.log_prob(branch_lengths)  # Call to ensure compilation
        self.grad(branch_lengths)

    def calculate_likelihoods(self, branch_lengths):
        return self.log_prob(branch_lengths)

    def calculate_branch_gradients(self, branch_lengths):
        return self.grad(branch_lengths)


class LibsbnRatioTransformBenchmarkable(RatioTransformBenchmarkable):
    def initialize(self, topology_file):
        self.inst = treeflow.libsbn.get_instance(topology_file)
        self.tree = self.inst.tree_collection.trees[0]
        self.node_height_state = np.array(self.tree.node_heights, copy=False)

        def libsbn_forward(ratios):
            self.tree.initialize_time_tree_using_height_ratios(ratios)
            return np.array(self.node_height_state[-ratios.shape[-1] :]).astype(
                ratios.dtype
            )

        self.forward_1d = libsbn_forward

        self.forward = np.vectorize(
            libsbn_forward, [treeflow.DEFAULT_FLOAT_DTYPE_NP], signature="(n)->(n)"
        )

        def libsbn_gradient(heights, dheights):
            self.node_height_state[-heights.shape[-1] :] = heights
            return np.array(
                libsbn.ratio_gradient_of_height_gradient(self.tree, dheights),
                dtype=heights.dtype,
            )

        self.grad_1d = libsbn_gradient

        self.grad = np.vectorize(
            libsbn_gradient, [treeflow.DEFAULT_FLOAT_DTYPE_NP], signature="(n),(n)->(n)"
        )

    def calculate_heights(self, ratios):
        return self.forward(ratios.numpy())

    def calculate_ratio_gradients(self, ratios, heights, height_gradients):
        return self.grad(heights.numpy(), height_gradients.numpy())
