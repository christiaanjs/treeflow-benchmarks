import tensorflow as tf
import numpy as np

from tensorflow.python.framework.tensor_conversion_registry import get
from treeflow_benchmarks.benchmarking import LikelihoodBenchmarkable
from treeflow_pipeline.model import get_likelihood


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
