import tensorflow as tf
import numpy as np

from treeflow_benchmarks.benchmarking import (
    LikelihoodBenchmarkable,
    RatioTransformBenchmarkable,
)
from treeflow_benchmarks.tree_transform import get_bij
from treeflow.acceleration.bito.beagle import phylogenetic_likelihood
from treeflow.acceleration.bito.ratio_transform import ratios_to_node_heights
from treeflow.acceleration.bito.instance import get_instance
from treeflow.model.phylo_model import (
    PhyloModel,
    get_subst_model,
    get_subst_model_params,
)
from treeflow_benchmarks.params import get_return_value_of_empty_generator


class BeagleLikelihoodBenchmarkable(LikelihoodBenchmarkable):
    phylo_model = None

    def initialize(
        self, topology_file, fasta_file, model, calculate_clock_rate_gradients
    ):

        phylo_model = PhyloModel(model)
        self.phylo_model = phylo_model
        self.calculate_clock_rate_gradients = calculate_clock_rate_gradients

        subst_model = get_subst_model(phylo_model.subst_model)
        subst_params, _ = get_return_value_of_empty_generator(
            get_subst_model_params(phylo_model.subst_model, phylo_model.subst_params)
        )
        log_prob, self.inst = phylogenetic_likelihood(
            fasta_file,
            subst_model,
            newick_file=topology_file,
            dated=True,
            site_model=phylo_model.site_model,
            site_model_params=phylo_model.site_params,
            **subst_params,
        )

        self.log_prob = tf.function(
            lambda branch_lengths, params: log_prob(branch_lengths)
        )

        def grad(branch_lengths, params):
            with tf.GradientTape() as t:
                t.watch(branch_lengths)
                log_prob_val = self.log_prob(branch_lengths, params)
            return t.gradient(log_prob_val, branch_lengths), None

        self.grad = tf.function(grad)

        branch_lengths = np.array(self.inst.tree_collection.trees[0].branch_lengths)[
            :-1
        ]
        self.log_prob(branch_lengths, None)  # Call to ensure compilation
        self.grad(branch_lengths, None)

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
