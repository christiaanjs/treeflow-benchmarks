import tensorflow as tf
import numpy as np
import treeflow
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
import treeflow_benchmarks.benchmarking as bench
from treeflow_benchmarks.treeflow import get_subst_model
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow_pipeline.model import cast
from treeflow.evolution.seqio import Alignment
from treeflow.tree.io import parse_newick
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from treeflow.evolution.substitution.probabilities import (
    get_batch_transition_probabilities_eigen,
)

DEFAULT_INT_DTYPE_TF = tf.int32


def move_axis_to_beginning(x, axis):
    rank = tf.rank(x)
    return tf.transpose(
        x,
        tf.concat(
            [[axis], tf.range(0, axis), tf.range(axis + 1, rank)],
            axis=0,
        ),
    )


class TensorflowLikelihood:
    def __init__(self, category_count=1, *args, **kwargs):
        self.category_count = category_count

    def set_topology(self, topology: TensorflowTreeTopology):
        self.taxon_count = topology.taxon_count
        self.node_indices_tensor = topology.postorder_node_indices
        self.child_indices_tensor = tf.cast(
            tf.gather(topology.child_indices, topology.postorder_node_indices),
            dtype=DEFAULT_INT_DTYPE_TF,
        )

        # preorder_indices = topology.preorder_indices[1:]
        # self.preorder_indices_tensor = tf.convert_to_tensor(
        #     preorder_indices, dtype=DEFAULT_INT_DTYPE_TF
        # )
        # self.preorder_sibling_indices_tensor = tf.cast(
        #     tf.gather(
        #         topology_dict["sibling_indices"],
        #         preorder_indices,
        #     ),
        #     dtype=DEFAULT_INT_DTYPE_TF,
        # )
        # self.preorder_parent_indices_tensor = tf.cast(
        #     tf.gather(
        #         topology_dict["parent_indices"],
        #         preorder_indices,
        #     ),
        #     dtype=DEFAULT_INT_DTYPE_TF,
        # )

    def get_vertex_count(self):
        return 2 * self.taxon_count - 1

    def init_postorder_partials(self, sequences_encoded, pattern_counts=None):
        """
        Sequence shape:
        # ..., pattern, taxon, character
        Partial shape:
        # Node, ..., category, pattern, character
        """
        self.taxon_count = sequences_encoded.shape[-2]
        self.pattern_count = sequences_encoded.shape[-3]
        self.pattern_counts = (
            tf.ones([self.pattern_count], dtype=sequences_encoded.dtype)
            if pattern_counts is None
            else pattern_counts
        )
        self.batch_shape = sequences_encoded.shape[:-3]
        character_shape = sequences_encoded.shape[-1]
        sequences_rank = tf.rank(sequences_encoded)
        sequences_encoded_taxon_first = move_axis_to_beginning(
            sequences_encoded, sequences_rank - 2
        )
        self.leaf_partials = tf.broadcast_to(
            tf.expand_dims(sequences_encoded_taxon_first, -2),  # Add category
            [self.taxon_count]
            + self.batch_shape
            + [self.pattern_count, self.category_count, character_shape],
        )

    def compute_postorder_partials(self, transition_probs):
        """
        transition_probs - [node, ..., category, parent char, child char]
        """
        self.postorder_partials_ta = tf.TensorArray(
            dtype=DEFAULT_FLOAT_DTYPE_TF,
            size=self.get_vertex_count(),
            element_shape=self.leaf_partials.shape[1:],
        )
        for i in range(self.taxon_count):
            self.postorder_partials_ta = self.postorder_partials_ta.write(
                i, self.leaf_partials[i]
            )

        child_transition_probs = tf.gather(transition_probs, self.child_indices_tensor)

        for i in tf.range(self.taxon_count - 1):
            node_index = self.node_indices_tensor[i]
            node_child_transition_probs = child_transition_probs[
                i
            ]  # child, ..., parent character, child character
            node_child_indices = self.child_indices_tensor[i]
            child_partials = self.postorder_partials_ta.gather(
                node_child_indices
            )  # Child, ..., category, pattern, child character
            parent_child_probs = tf.expand_dims(
                node_child_transition_probs, -4
            ) * tf.expand_dims(  # child, ..., category, pattern, parent char, child char
                child_partials, -2
            )
            node_partials = tf.reduce_prod(
                tf.reduce_sum(
                    parent_child_probs,
                    axis=-1,
                ),
                axis=0,
            )
            self.postorder_partials_ta = self.postorder_partials_ta.write(
                node_index, node_partials
            )

    def compute_likelihood_from_partials(self, freqs, category_weights):
        root_partials = self.postorder_partials_ta.gather([2 * self.taxon_count - 2])[0]
        cat_likelihoods = tf.reduce_sum(freqs * root_partials, axis=-1)
        site_likelihoods = tf.reduce_sum(category_weights * cat_likelihoods, axis=-1)
        return tf.reduce_sum(
            tf.math.log(site_likelihoods) * self.pattern_counts, axis=-1
        )


class TreeflowOldLikelihoodBenchmarkable(bench.LikelihoodBenchmarkable):
    def initialize_tf_log_prob(
        self,
        fasta_file,
        model,
    ):
        if model.site_model == "none":
            category_count = 1
            category_weights = cast([1.0])
            category_rates = cast([1.0])
        else:
            raise ValueError(f"Unknown site model: {model.site_model}")

        alignment = Alignment(fasta_file).get_compressed_alignment()

        subst_model, subst_model_params = get_subst_model(model)
        self._likelihood = TensorflowLikelihood(category_count)
        self._likelihood.set_topology(self.tree.topology)
        self._likelihood.init_postorder_partials(
            alignment.get_encoded_sequence_tensor(self.tree.taxon_set),
            pattern_counts=alignment.get_weights_tensor(),
        )
        eigen = subst_model.eigen(**subst_model_params)

        def log_prob(branch_lengths):
            with_categories = tf.expand_dims(branch_lengths, 1) * tf.expand_dims(
                category_rates, 0
            )  # node, category
            transition_probs = get_batch_transition_probabilities_eigen(
                eigen, with_categories, batch_rank=2
            )
            self._likelihood.compute_postorder_partials(transition_probs)
            return self._likelihood.compute_likelihood_from_partials(
                subst_model_params["frequencies"], category_weights
            )

        return log_prob

    def initialize(self, topology_file, fasta_file, model):
        self.tree = convert_tree_to_tensor(parse_newick(topology_file))
        log_prob = self.initialize_tf_log_prob(fasta_file, model)
        self.log_prob_1d = log_prob

        # def log_prob_vectorised(branch_lengths):
        #     return tf.map_fn(self.log_prob_1d, branch_lengths)

        # self.log_prob = tf.function(log_prob_vectorised)
        self.log_prob = tf.function(log_prob)

        def grad(branch_lengths):
            with tf.GradientTape() as t:
                t.watch(branch_lengths)
                val = self.log_prob(branch_lengths)
                return t.gradient(val, branch_lengths)

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
