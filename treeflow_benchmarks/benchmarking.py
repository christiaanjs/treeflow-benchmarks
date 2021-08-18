from abc import abstractmethod
from collections import namedtuple
import tensorflow as tf
import treeflow.sequences
from timeit import default_timer as timer


def time_function(func, *args, **kwargs):
    start = timer()
    res = func(*args, **kwargs)
    stop = timer()
    return stop - start, res


def get_class_with_metadata(_class):
    return namedtuple(
        f"{_class.__name__}WithMetadata",
        _class._fields + ("taxon_count", "seed", "method"),
    )


LikelihoodTimes = namedtuple("LikelihoodTimes", ["likelihood_time", "gradient_time"])
LikelihoodTimesWithMetadata = get_class_with_metadata(LikelihoodTimes)
RatioTransformTimes = namedtuple(
    "RatioTransformTimes", ["forward_time", "gradient_time"]
)
RatioTransformTimesWithMetadata = get_class_with_metadata(RatioTransformTimes)
CoalescentTimes = namedtuple(
    "CoalescentTimes",
    ["likelihood_time", "height_gradient_time", "pop_size_gradient_time"],
)
CoalescentTimesWithMetadata = get_class_with_metadata(CoalescentTimes)

types_with_metadata = {
    LikelihoodTimes: LikelihoodTimesWithMetadata,
    RatioTransformTimes: RatioTransformTimesWithMetadata,
    CoalescentTimes: CoalescentTimesWithMetadata,
}


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


def benchmark_likelihood(topology_file, fasta_file, model, trees, benchmarkable):
    benchmarkable.initialize(topology_file, fasta_file, model)

    branch_lengths = treeflow.sequences.get_branch_lengths(trees).numpy()

    likelihood_time, likelihood_res = time_function(
        benchmarkable.calculate_likelihoods, branch_lengths
    )

    gradient_time, gradient_res = time_function(
        benchmarkable.calculate_branch_gradients, branch_lengths
    )

    return LikelihoodTimes(likelihood_time, gradient_time)


def annotate_times(times, taxon_count, seed, method):
    return types_with_metadata[type(times)](
        taxon_count=taxon_count, seed=seed, method=method, **times._asdict()
    )


class RatioTransformBenchmarkable:
    @abstractmethod
    def initialize(self, topology_file):
        pass

    @abstractmethod
    def calculate_heights(self, ratios):
        pass

    @abstractmethod
    def calculate_ratio_gradients(self, ratios, heights, height_gradients):
        pass


def benchmark_ratio_transform(topology_file, ratios, trees, benchmarkable):
    benchmarkable.initialize(topology_file)

    taxon_count = ratios.shape[-1] + 1
    heights = trees["heights"][..., taxon_count:]

    forward_time, forward_res = time_function(benchmarkable.calculate_heights, ratios)
    gradient_time, gradient_res = time_function(
        benchmarkable.calculate_ratio_gradients,
        ratios,
        heights,
        tf.ones_like(ratios),  # TODO: Non-dummy gradient value
    )
    return RatioTransformTimes(forward_time, gradient_time)


class CoalescentPriorBenchmarkable:
    @abstractmethod
    def initialize(self, topology_file, pop_size):
        pass

    @abstractmethod
    def calculate_likelihood(self, trees):
        pass

    @abstractmethod
    def calculate_height_gradients(self, trees):
        pass

    @abstractmethod
    def calculate_pop_size_gradients(self, trees):
        pass
