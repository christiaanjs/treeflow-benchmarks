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
        inst = treeflow.libsbn.get_instance(topology_file)
        self.bij, self.taxon_count = get_bij(
            topology_file, treeflow.tree_transform.Ratio, inst=inst
        )
        self.forward = tf.function(self.bij.forward)

        def grad(ratios, height_gradients):
            with tf.GradientTape() as t:
                t.watch(ratios)
                heights = self.bij.forward(ratios)
                return t.gradient(heights, ratios, output_gradients=height_gradients)

        self.grad = tf.function(grad)

        tree, _ = treeflow.tree_processing.parse_newick(topology_file)
        ratios = self.bij.inverse(tree["heights"][self.taxon_count :])
        self.forward(ratios)  # Do tracing
        self.grad(ratios, tf.ones_like(ratios))

    def calculate_heights(self, ratios):
        return self.forward(ratios)

    def calculate_ratio_gradients(self, ratios, height_gradients):
        return self.grad(ratios, height_gradients)
