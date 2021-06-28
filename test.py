import pathlib

from treeflow_pipeline.util import (
    text_input,
    text_output,
    yaml_input,
    yaml_output,
    pickle_input,
    pickle_output,
)
import treeflow_pipeline.simulation as pipe_sim
import treeflow_pipeline.templating as tem
import treeflow_pipeline.model as mod
import treeflow_benchmarks.simulation as bench_sim
import treeflow_pipeline.topology_inference as top
import treeflow_benchmarks.benchmarking as bench
import treeflow_benchmarks.tf_likelihood as bench_tf
import pandas as pd
import treeflow_benchmarks.libsbn

likelihood_benchmarkables = dict(
    tensorflow=bench_tf.TensorflowLikelihoodBenchmarkable(custom_gradient=False),
    tensorflow_custom_gradient=bench_tf.TensorflowLikelihoodBenchmarkable(
        custom_gradient=True
    ),
)
res = bench.benchmark_likelihood(
    "out/20taxa/1seed/topology-sim.newick",
    "out/20taxa/1seed/sequences.fasta",
    mod.Model(yaml_input("model.yaml")),
    pickle_input("out/20taxa/1seed/tree-sim.pickle"),
    treeflow_benchmarks.libsbn.BeagleLikelihoodBenchmarkable(),
)
print(res)
