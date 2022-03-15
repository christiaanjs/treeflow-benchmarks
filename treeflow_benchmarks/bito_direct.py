import numpy as np
from treeflow_benchmarks.bito import BeagleLikelihoodBenchmarkable


class BeagleDirectLikelihoodBenchmarkable(BeagleLikelihoodBenchmarkable):
    def initialize(self, topology_file, fasta_file, model):
        super().initialize(topology_file, fasta_file, model)
        self.branch_length_state = np.array(
            self.inst.tree_collection.trees[0].branch_lengths, copy=False
        )

    def calculate_likelihoods(self, branch_lengths: np.ndarray) -> np.ndarray:
        # batch_size = branch_lengths.shape[0]
        # output = np.zeros(batch_size, dtype=branch_lengths.dtype)
        # for i in range(batch_size):
        self.branch_length_state[:-1] = branch_lengths  # [i]
        res = np.array(self.inst.log_likelihoods())[0]
        # output[i] = res
        return res

    def calculate_branch_gradients(self, branch_lengths: np.ndarray) -> np.ndarray:
        # output = np.zeros_like(branch_lengths)
        # for i in range(branch_lengths.shape[0]):
        self.branch_length_state[:-1] = branch_lengths
        gradient = self.inst.phylo_gradients()[0]
        gradient_array = np.array(gradient.gradient["branch_lengths"])
        output = gradient_array[:-1]
        return output
