from treeflow_benchmarks.jax import JaxLikelihoodBenchmarkable
import pytest

from treeflow_benchmarks.params import get_numpy_gradient_params_dict
from treeflow.model.phylo_model import PhyloModel
import tensorflow as tf

@pytest.mark.parametrize("calculate_clock_rate_gradient", [True, False])
@pytest.mark.parametrize("jit", [False, True])
def test_jax_likelihood(
    topology_file, fasta_file, branch_lengths, model, jit, calculate_clock_rate_gradient
):
    benchmarkable = JaxLikelihoodBenchmarkable(jit=jit)
    benchmarkable.initialize(
        topology_file, fasta_file, model, calculate_clock_rate_gradient
    )
    params = get_numpy_gradient_params_dict(
        PhyloModel(model), calculate_clock_rate_gradient=calculate_clock_rate_gradient
    )
    likelihood_res = benchmarkable.calculate_likelihoods_loop(branch_lengths, params)

    assert likelihood_res.shape == (branch_lengths.shape[0],)
    gradient_res = benchmarkable.calculate_gradients_loop(branch_lengths, params)

    def assert_shape(res, param):
        assert res.shape == (branch_lengths.shape[0],) + param.shape

    tf.nest.map_structure(assert_shape, gradient_res, [branch_lengths[0], params])
