from logging import log
import treeflow_benchmarks.benchmarking as bench
import treeflow.tree_processing
import treeflow.sequences
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
        tree, taxon_names = treeflow.tree_processing.parse_newick(topology_file)
        topology_dict = treeflow.tree_processing.update_topology_dict(tree["topology"])
        alignment = treeflow.sequences.get_encoded_sequences(fasta_file, taxon_names)
        subst_model = subst_model_classes[model.subst_model](**model.subst_params)
        if model.site_model == "none":
            category_weights = np.array([1.0])
            category_rates = np.array([1.0])
        else:
            raise ValueError(f"Unknown site model: {model.site_model}")
        self.jax_likelihood = phylojax.likelihood.JaxLikelihood(
            topology_dict,
            alignment["sequences"].numpy(),
            subst_model,
            alignment["weights"],
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

        branch_lengths = treeflow.sequences.get_branch_lengths(tree).numpy()
        self.log_prob(branch_lengths)
        self.grad(branch_lengths)

    def calculate_likelihoods(self, branch_lengths):
        return self.log_prob(branch_lengths)

    def calculate_branch_gradients(self, branch_lengths):
        return self.grad(branch_lengths)
