import tensorflow as tf
import treeflow
import numpy as np
from treeflow_pipeline.model import cast
import treeflow_benchmarks.benchmarking as bench
from treeflow.evolution.substitution.nucleotide.hky import HKY

subst_model_classes = dict(hky=HKY)


def get_subst_model(model):
    subst_model = subst_model_classes[model.subst_model]()
    subst_model_params = {key: cast(value) for key, value in model.subst_params.items()}
    return subst_model, subst_model_params


class TreeflowLikelihoodBenchmarkable(bench.LikelihoodBenchmarkable):
    log_prob = None
    grad = None
    tree = None
    taxon_names = None
