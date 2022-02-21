from functools import wraps
from treeflow_benchmarks.treeflow_old import TreeflowOldLikelihoodBenchmarkable
from treeflow_benchmarks.tree_transform import TensorflowRatioTransformBenchmarkable
from treeflow_benchmarks.jax import JaxLikelihoodBenchmarkable
from treeflow_benchmarks.bito import (
    BeagleLikelihoodBenchmarkable,
    BitoRatioTransformBenchmarkable,
)
from treeflow_benchmarks.benchmarking import (
    benchmark_likelihood,
    benchmark_ratio_transform,
    LikelihoodTimes,
    RatioTransformTimes,
    CoalescentTimes,
)

likelihood_benchmarkables = dict(
    treeflow_old=TreeflowOldLikelihoodBenchmarkable(),
    beagle_bito=BeagleLikelihoodBenchmarkable(),
    jax=JaxLikelihoodBenchmarkable(),
    jax_jit=JaxLikelihoodBenchmarkable(jit=True),
)

ratio_transform_benchmarkables = dict(
    tensorflow=TensorflowRatioTransformBenchmarkable(),
    beagle_bito=BitoRatioTransformBenchmarkable(),
)

benchmarkables = dict(
    likelihood=likelihood_benchmarkables,
    ratio_transform=ratio_transform_benchmarkables,
)

output_types = dict(
    likelihood=LikelihoodTimes,
    ratio_transform=RatioTransformTimes,
    coalescent=CoalescentTimes,
)


def wrap_benchmark_func(func, *sim_keys):
    @wraps(func)
    def benchmark_func(benchmarkable, **sim_kwargs):
        return func(
            benchmarkable=benchmarkable,
            **{key: value for key, value in sim_kwargs.items() if key in sim_keys}
        )

    return benchmark_func


benchmark_functions = dict(
    likelihood=wrap_benchmark_func(
        benchmark_likelihood, "topology_file", "fasta_file", "model", "trees"
    ),
    ratio_transform=wrap_benchmark_func(
        benchmark_ratio_transform, "topology_file", "ratios", "trees", "benchmarkable"
    ),
)
