from abc import abstractmethod


class LikelihoodBenchmarkable:
    @abstractmethod
    def initialize(self, topology_file, fasta_file, model):
        pass

    @abstractmethod
    def calculate_likelihoods(self, branch_lengths):
        pass

    @abstractmethod
    def calculate_branch_gradients(self, branch_lengths):
        pass


def benchmark_likelihood(topology_file, fasta_file, model, trees):
    pass
