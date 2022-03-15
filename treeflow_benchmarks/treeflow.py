from lib2to3.pgen2.tokenize import untokenize
import tensorflow as tf
import treeflow
import numpy as np
from treeflow_pipeline.model import cast
import treeflow_benchmarks.benchmarking as bench
from treeflow.evolution.substitution.nucleotide.hky import HKY
from treeflow.evolution.seqio import Alignment
from treeflow.tree.io import parse_newick
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from treeflow.evolution.substitution.probabilities import (
    get_transition_probabilities_tree_eigen,
)
from treeflow.distributions.leaf_ctmc import LeafCTMC
from treeflow.distributions.sample_weighted import SampleWeighted

subst_model_classes = dict(hky=HKY)


def get_subst_model(model):
    subst_model = subst_model_classes[model.subst_model]()
    subst_model_params = {key: cast(value) for key, value in model.subst_params.items()}
    return subst_model, subst_model_params


class TreeflowLikelihoodBenchmarkable(bench.LikelihoodBenchmarkable):
    log_prob = None
    grad = None
    tree = None
    taxon_names = None

    def initialize(self, topology_file, fasta_file, model):
        self.tree = convert_tree_to_tensor(parse_newick(topology_file))
        alignment = Alignment(fasta_file).get_compressed_alignment()
        weights = alignment.get_weights_tensor()
        sequences_encoded = alignment.get_encoded_sequence_tensor(self.tree.taxon_set)
        site_count = alignment.site_count

        subst_model, subst_model_params = get_subst_model(model)
        subst_model_params = {
            key: tf.convert_to_tensor(value, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF)
            for key, value in subst_model_params.items()
        }
        self.eigen = subst_model.eigen(**subst_model_params)

        if model.site_model != "none":
            raise ValueError(f"Unknown site model: {model.site_model}")

        unrooted_tree = self.tree.get_unrooted_tree()

        def log_prob_1d(branch_lengths):
            tree = unrooted_tree.with_branch_lengths(branch_lengths)
            transition_probs_tree = get_transition_probabilities_tree_eigen(
                tree, self.eigen
            )
            dist = SampleWeighted(
                LeafCTMC(transition_probs_tree, subst_model_params["frequencies"]),
                weights=weights,
                sample_shape=(site_count,),
            )
            return dist.log_prob(sequences_encoded)

        # def log_prob(branch_lengths):
        #     return tf.map_fn(log_prob_1d, branch_lengths)

        self.log_prob = tf.function(log_prob_1d)

        def grad(branch_lengths):
            with tf.GradientTape() as t:
                t.watch(branch_lengths)
                log_prob_val = self.log_prob(branch_lengths)
            return t.gradient(log_prob_val, branch_lengths)

        self.grad = tf.function(grad)

        branch_lengths = self.tree.branch_lengths
        # self.log_prob(tf.expand_dims(branch_lengths, 0))  # Call to ensure compilation
        # self.grad(tf.expand_dims(branch_lengths, 0))
        self.log_prob(branch_lengths)  # Call to ensure compilation
        self.grad(branch_lengths)

    def calculate_likelihoods(self, branch_lengths: np.ndarray) -> np.ndarray:
        return self.log_prob(
            tf.convert_to_tensor(branch_lengths, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF)
        ).numpy()

    def calculate_branch_gradients(self, branch_lengths: np.ndarray) -> np.ndarray:
        return self.grad(
            tf.convert_to_tensor(branch_lengths, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF)
        ).numpy()
