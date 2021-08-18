from treeflow_benchmarks.tf_likelihood import TensorflowLikelihoodBenchmarkable
from treeflow_benchmarks.tree_transform import TensorflowRatioTransformBenchmarkable
from treeflow_benchmarks.jax import JaxLikelihoodBenchmarkable
from treeflow_benchmarks.libsbn import (
    BeagleLikelihoodBenchmarkable,
    LibsbnRatioTransformBenchmarkable,
)

likelihood_benchmarkables = dict(
    tensorflow=TensorflowLikelihoodBenchmarkable(custom_gradient=False),
    beagle=BeagleLikelihoodBenchmarkable(),
    jax=JaxLikelihoodBenchmarkable(),
)

ratio_transform_benchmarkables = dict(
    tensorflow=TensorflowRatioTransformBenchmarkable(),
    libsbn=LibsbnRatioTransformBenchmarkable(),
)
