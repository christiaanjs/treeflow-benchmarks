import tensorflow as tf
import numpy as np

from treeflow_benchmarks.benchmarking import (
    LikelihoodBenchmarkable,
    RatioTransformBenchmarkable,
)
from treeflow_benchmarks.tree_transform import get_bij
from treeflow_benchmarks.treeflow import get_subst_model
from treeflow.acceleration.bito.beagle import phylogenetic_likelihood
from treeflow.acceleration.bito.ratio_transform import ratios_to_node_heights
from treeflow.acceleration.bito.instance import get_instance


class BeagleLikelihoodBenchmarkable(LikelihoodBenchmarkable):
    def initialize(self, topology_file, fasta_file, model):
        def grad(branch_lengths):
            with tf.GradientTape() as t:
                t.watch(branch_lengths)
                val = self.log_prob(branch_lengths)
                return t.gradient(val, branch_lengths)

        subst_model, subst_model_params = get_subst_model(model)
        log_prob, self.inst = phylogenetic_likelihood(
            fasta_file,
            subst_model,
            newick_file=topology_file,
            dated=True,
            **subst_model_params
        )

        self.log_prob = tf.function(log_prob)

        def grad(branch_lengths):
            with tf.GradientTape() as t:
                t.watch(branch_lengths)
                log_prob_val = self.log_prob(branch_lengths)
            return t.gradient(log_prob_val, branch_lengths)

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


class BitoRatioTransformBenchmarkable(RatioTransformBenchmarkable):
    def initialize(self, topology_file):
        self.inst = get_instance(topology_file, dated=True)
        self.tree = self.inst.tree_collection.trees[0]
        self.node_height_state = np.array(self.tree.node_heights, copy=False)

    def forward(self, ratios):
        raise NotImplemented("Bito ratio transform not yet implemented")

    def grad(self, heights, dheights):
        raise NotImplemented("Bito ratio transform not yet implemented")

    def calculate_heights(self, ratios):
        return self.forward(ratios.numpy())

    def calculate_ratio_gradients(self, ratios, heights, height_gradients):
        return self.grad(heights.numpy(), height_gradients.numpy())
