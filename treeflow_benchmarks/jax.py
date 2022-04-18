from logging import log
import treeflow_benchmarks.benchmarking as bench
from treeflow.tree.io import parse_newick
from treeflow.evolution.seqio import Alignment
import phylojax.likelihood
import phylojax.substitution
import phylojax.site_rate_variation
import jax.numpy as np
import jax
from treeflow.model.phylo_model import PhyloModel
from treeflow_benchmarks.params import get_numpy_gradient_params_dict

subst_model_classes = dict(
    hky=phylojax.substitution.HKY,
    jc=phylojax.substitution.JC,
    gtr=phylojax.substitution.GTR,
)


class JaxLikelihoodBenchmarkable(bench.LikelihoodBenchmarkable):
    def __init__(self, jit=False):
        self.jit = jit
        self.jax_likelihood = None
        self.log_prob = None

    def initialize(
        self, topology_file, fasta_file, model, calculate_clock_rate_gradient
    ):
        phylo_model = PhyloModel(model)
        params = get_numpy_gradient_params_dict(
            phylo_model, calculate_clock_rate_gradient
        )
        tree = parse_newick(topology_file)
        compressed_alignment = Alignment(fasta_file).get_compressed_alignment()
        encoded_sequences = compressed_alignment.get_encoded_sequence_array(
            tree.taxon_set
        )
        encoded_sequences_node_first = np.moveaxis(encoded_sequences, 1, 0)
        pattern_counts = compressed_alignment.get_weights_array()
        DEFAULT_WEIGHTS = np.array([1.0])
        DEFAULT_RATES = np.array([1.0])
        topology_dict = dict(
            child_indices=tree.topology.child_indices,
            postorder_node_indices=tree.topology.postorder_node_indices,
        )

        def log_prob(branch_lengths, params):
            if phylo_model.site_model == "discrete_gamma":
                (
                    category_weights,
                    category_rates,
                ) = phylojax.site_rate_variation.get_discrete_gamma_weights_rates(
                    **params["site_model_params"],
                    category_count=phylo_model.site_params["category_count"],
                )
            elif phylo_model.site_model == "discrete_weibull":
                (
                    category_weights,
                    category_rates,
                ) = phylojax.site_rate_variation.get_discrete_weibull_weights_rates(
                    **params["site_model_params"],
                    category_count=phylo_model.site_params["category_count"],
                )
            elif phylo_model.site_model == "none":
                category_weights = DEFAULT_WEIGHTS
                category_rates = DEFAULT_RATES
            else:
                raise ValueError(f"Unknown site model: {phylo_model.site_model}")

            subst_model = subst_model_classes[phylo_model.subst_model](
                **params["subst_model_params"]
            )

            if phylo_model.clock_model == "strict":
                if calculate_clock_rate_gradient:
                    rates = params["clock_model_params"]["clock_rate"]
                else:
                    rates = phylo_model.clock_params["clock_rate"]
            else:
                raise ValueError(f"Unknown clock model: {phylo_model.clock_model}")
            likelihood = phylojax.likelihood.JaxLikelihood(
                topology_dict,
                encoded_sequences_node_first,
                subst_model,
                pattern_counts,
                category_weights,
                category_rates,
            )
            scaled_branch_lengths = branch_lengths * rates

            return likelihood.log_likelihood(scaled_branch_lengths)

        grad = lambda *args: list(jax.grad(log_prob, argnums=[0, 1])(*args))

        if self.jit:
            self.log_prob = jax.jit(log_prob)
            self.grad = jax.jit(grad)
        else:
            self.log_prob = log_prob
            self.grad = grad

        branch_lengths = tree.branch_lengths
        self.log_prob(branch_lengths, params)
        self.grad(branch_lengths, params)

    def calculate_likelihoods(self, branch_lengths, params):
        return self.log_prob(branch_lengths, params)

    def calculate_gradients(self, branch_lengths, params):
        return self.grad(branch_lengths, params)
