from abc import abstractmethod
from collections import namedtuple
import treeflow.sequences
from timeit import default_timer as timer


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


LikelihoodTimes = namedtuple("LikelihoodTimes", ["likelihood_time", "gradient_time"])


def time_function(func, *args, **kwargs):
    start = timer()
    res = func(*args, **kwargs)
    stop = timer()
    return stop - start, res


def benchmark_likelihood(topology_file, fasta_file, model, trees, benchmarkable):
    benchmarkable.initialize(topology_file, fasta_file, model)

    branch_lengths = treeflow.sequences.get_branch_lengths(trees)

    likelihood_time, likelihood_res = time_function(
        benchmarkable.calculate_likelihoods, branch_lengths
    )

    gradient_time, gradient_res = time_function(
        benchmarkable.calculate_branch_gradients, branch_lengths
    )

    return LikelihoodTimes(likelihood_time, gradient_time)


LikelihoodTimesWithMetadata = namedtuple(
    "LikelihoodTimesWithMetadata",
    LikelihoodTimes._fields + ("taxon_count", "seed", "likelihood"),
)

types_with_metadata = {LikelihoodTimes: LikelihoodTimesWithMetadata}


def annotate_times(times, taxon_count, seed, likelihood):
    return types_with_metadata[type(times)](
        taxon_count=taxon_count, seed=seed, likelihood=likelihood, **times._asdict()
    )
