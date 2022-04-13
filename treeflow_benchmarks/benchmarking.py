import typing as tp
from abc import abstractmethod
from collections import namedtuple
import tensorflow as tf
from timeit import default_timer as timer
import numpy as np
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.model.phylo_model import PhyloModel
from treeflow_benchmarks.params import get_numpy_gradient_params_dict


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


LikelihoodTimes = namedtuple(
    "LikelihoodTimes", ["likelihood_time", "branch_gradient_time"]
)
LikelihoodTimesWithMetadata = get_class_with_metadata(LikelihoodTimes)
RatioTransformTimes = namedtuple(
    "RatioTransformTimes", ["forward_time", "ratio_gradient_time"]
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
    def initialize(
        self, topology_file, fasta_file, model, calculate_clock_rate_gradient
    ):
        pass

    @abstractmethod
    def calculate_likelihoods(
        self, branch_lengths: np.ndarray, params: object
    ) -> np.ndarray:
        pass

    @abstractmethod
    def calculate_gradients(
        self, branch_lengths: np.ndarray, params: object
    ) -> np.ndarray:
        pass

    def calculate_likelihoods_loop(self, branch_lengths: np.ndarray, params: object):
        batch_size = branch_lengths.shape[0]
        output = np.zeros(batch_size, dtype=branch_lengths.dtype)
        for i in range(batch_size):
            output[i] = self.calculate_likelihoods(branch_lengths[i], params)
        return output

    def calculate_gradients_loop(self, branch_lengths: np.ndarray, params: object):
        batch_shape = branch_lengths.shape[:-1]
        batch_size = batch_shape[0]
        structure = [branch_lengths[0], params]
        output = tf.nest.map_structure(
            lambda x: np.zeros(batch_shape + x.shape, x.dtype), structure
        )
        for i in range(batch_size):
            gradients = self.calculate_gradients(branch_lengths[i], params)
            tf.nest.map_structure(lambda out, grad: out.put(i, grad), output, gradients)
        return output


def benchmark_likelihood(
    topology_file,
    fasta_file,
    model,
    trees: tp.Union[NumpyRootedTree, TensorflowRootedTree],
    benchmarkable: LikelihoodBenchmarkable,
):
    benchmarkable.initialize(
        topology_file, fasta_file, model, True
    )  # TODO: Separate models

    trees = trees.numpy() if isinstance(trees, TensorflowRootedTree) else trees
    branch_lengths = trees.branch_lengths
    params = get_numpy_gradient_params_dict(PhyloModel(model))

    likelihood_time, likelihood_res = time_function(
        benchmarkable.calculate_likelihoods_loop, branch_lengths, params
    )

    gradient_time, gradient_res = time_function(
        benchmarkable.calculate_gradients_loop, branch_lengths, params
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


def benchmark_ratio_transform(
    topology_file, ratios, trees: NumpyRootedTree, benchmarkable
):
    benchmarkable.initialize(topology_file)
    taxon_count = trees.taxon_count
    heights = trees.node_heights

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
