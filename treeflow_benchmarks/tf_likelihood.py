import tensorflow as tf
from treeflow_pipeline.model import cast
import treeflow.sequences
import treeflow.substitution_model
import treeflow.tree_processing
import treeflow_benchmarks.benchmarking as bench

subst_model_classes = dict(hky=treeflow.substitution_model.HKY)


class TensorflowLikelihoodBenchmarkable(bench.LikelihoodBenchmarkable):
    log_prob = None
    grad = None
    tree = None
    taxon_names = None

    def initialize_tf_log_prob(
        self,
        topology_file,
        fasta_file,
        model,
    ):
        if model.site_model == "none":
            category_count = 1
            category_weights = cast([1.0])
            category_rates = cast([1.0])
        else:
            raise ValueError(f"Unknown site model: {model.site_model}")

        alignment = treeflow.sequences.get_encoded_sequences(
            fasta_file, self.taxon_names
        )

        subst_model = subst_model_classes[model.subst_model]()
        subst_model_params = {
            key: cast(value) for key, value in model.subst_params.items()
        }
        return treeflow.sequences.log_prob_conditioned_branch_only(
            alignment,
            self.tree["topology"],
            category_count,
            subst_model,
            category_weights,
            category_rates,
            custom_gradient=False,
            **subst_model_params,
        )

    def initialize(self, topology_file, fasta_file, model):
        self.tree, self.taxon_names = treeflow.tree_processing.parse_newick(
            topology_file
        )
        log_prob = self.initialize_tf_log_prob(topology_file, fasta_file, model)
        self.log_prob = tf.function(log_prob)

        @tf.function
        def grad(branch_lengths):
            return tf.gradients(self.log_prob(branch_lengths), branch_lengths)

        self.grad = grad

        branch_lengths = treeflow.sequences.get_branch_lengths(self.tree)
        self.log_prob(branch_lengths)  # Call to ensure compilation
        self.grad(branch_lengths)

    def calculate_likelihoods(self, branch_lengths):
        return self.log_prob(branch_lengths)

    def calculate_branch_gradients(self, branch_lengths):
        return self.grad(branch_lengths)
