from heapq import merge
import typing as tp
import tensorflow as tf
import treeflow
import numpy as np
import treeflow_benchmarks.benchmarking as bench
from treeflow_benchmarks.params import (
    split_gradient_params,
    get_return_value_of_empty_generator,
    merge_params,
)
from treeflow.evolution.seqio import Alignment
from treeflow.tree.io import parse_newick
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from treeflow.model.phylo_model import (
    PhyloModel,
    get_sequence_distribution,
    get_subst_model,
    get_clock_model_rates,
)


class TreeflowLikelihoodBenchmarkable(bench.LikelihoodBenchmarkable):
    log_prob = None
    grad = None
    tree = None
    taxon_names = None

    def initialize(
        self, topology_file, fasta_file, model, calculate_clock_rate_gradient
    ):
        self.tree = convert_tree_to_tensor(parse_newick(topology_file))
        unrooted_tree = self.tree.get_unrooted_tree()
        alignment = Alignment(fasta_file).get_compressed_alignment()
        sequences_encoded = alignment.get_encoded_sequence_tensor(self.tree.taxon_set)

        phylo_model = PhyloModel(model)
        subst_model = get_subst_model(phylo_model.subst_model)
        self.gradient_params, self.non_gradient_params = split_gradient_params(
            phylo_model, calculate_clock_rate_gradient
        )
        self.params = merge_params(self.gradient_params, self.non_gradient_params)

        def log_prob(branch_lengths, gradient_params):
            tree = unrooted_tree.with_branch_lengths(branch_lengths)
            params = merge_params(gradient_params, self.non_gradient_params)
            subst_model_params = params["subst_model_params"]
            site_model_params = params["site_model_params"]
            clock_model_params = params["clock_model_params"]
            clock_model_rates = get_return_value_of_empty_generator(
                get_clock_model_rates(
                    phylo_model.clock_model, clock_model_params, True, self.tree
                )
            )
            sequence_dist = get_sequence_distribution(
                alignment,
                tree,
                subst_model,
                subst_model_params,
                phylo_model.site_model,
                site_model_params,
                clock_model_rates,
            )
            return sequence_dist.log_prob(sequences_encoded)

        self.log_prob = tf.function(log_prob)

        def grad(branch_lengths, params):
            with tf.GradientTape() as t:
                t.watch(branch_lengths)
                tf.nest.map_structure(t.watch, params)
                log_prob_val = self.log_prob(branch_lengths, params)
            return t.gradient(log_prob_val, [branch_lengths, params])

        self.grad = tf.function(grad)

        branch_lengths = self.tree.branch_lengths
        # self.log_prob(tf.expand_dims(branch_lengths, 0))  # Call to ensure compilation
        # self.grad(tf.expand_dims(branch_lengths, 0))

        self.log_prob(
            branch_lengths, self.gradient_params
        )  # Call to ensure compilation
        self.grad(branch_lengths, self.gradient_params)

    def calculate_gradients(
        self, branch_lengths: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        gradient_params_tensor = tf.nest.map_structure(
            lambda x, y: tf.constant(x, dtype=y.dtype), params, self.gradient_params
        )
        branch_lengths_tensor = tf.constant(
            branch_lengths, dtype=self.tree.heights.dtype
        )
        tensor_grad = self.grad(branch_lengths_tensor, gradient_params_tensor)
        return tf.nest.map_structure(lambda x: x.numpy(), tensor_grad)

    def calculate_likelihoods(
        self, branch_lengths: np.ndarray, params: object
    ) -> np.ndarray:
        gradient_params_tensor = tf.nest.map_structure(
            lambda x, y: tf.constant(x, dtype=y.dtype), params, self.gradient_params
        )
        branch_lengths_tensor = tf.constant(
            branch_lengths, dtype=self.tree.heights.dtype
        )
        return self.log_prob(branch_lengths_tensor, gradient_params_tensor).numpy()
