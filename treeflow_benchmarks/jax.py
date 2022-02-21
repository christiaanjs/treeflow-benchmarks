from logging import log
import treeflow_benchmarks.benchmarking as bench
from treeflow.tree.io import parse_newick
from treeflow.evolution.seqio import Alignment
import phylojax.likelihood
import phylojax.substitution
import jax.numpy as np
import jax

subst_model_classes = dict(hky=phylojax.substitution.HKY)


class JaxLikelihoodBenchmarkable(bench.LikelihoodBenchmarkable):
    def __init__(self, jit=False):
        self.jit = jit
        self.jax_likelihood = None
        self.log_prob = None

    def initialize(self, topology_file, fasta_file, model):
        tree = parse_newick(topology_file)
        compressed_alignment = Alignment(fasta_file).get_compressed_alignment()
        encoded_sequences = compressed_alignment.get_encoded_sequence_array(
            tree.taxon_set
        )
        subst_model = subst_model_classes[model.subst_model](**model.subst_params)
        if model.site_model == "none":
            category_weights = np.array([1.0])
            category_rates = np.array([1.0])
        else:
            raise ValueError(f"Unknown site model: {model.site_model}")
        self.jax_likelihood = phylojax.likelihood.JaxLikelihood(
            dict(
                child_indices=tree.topology.child_indices,
                postorder_node_indices=tree.topology.postorder_node_indices,
            ),
            encoded_sequences,
            subst_model,
            compressed_alignment.get_weights_array(),
            category_weights,
            category_rates,
        )

        def log_prob(branch_lengths):
            return self.jax_likelihood.log_likelihood(branch_lengths)

        grad = np.vectorize(jax.grad(log_prob), signature="(n)->(n)")

        if self.jit:
            self.log_prob = jax.jit(log_prob)
            self.grad = jax.jit(grad)
        else:
            self.log_prob = log_prob
            self.grad = grad

        branch_lengths = tree.branch_lengths
        self.log_prob(branch_lengths)
        self.grad(branch_lengths)

    def calculate_likelihoods(self, branch_lengths):
        return self.log_prob(branch_lengths)

    def calculate_branch_gradients(self, branch_lengths):
        return self.grad(branch_lengths)
