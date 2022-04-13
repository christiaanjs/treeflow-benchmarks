import pytest
import tensorflow as tf
from treeflow_benchmarks.bito_direct import BeagleDirectLikelihoodBenchmarkable
from treeflow_benchmarks.params import get_numpy_gradient_params_dict
from treeflow.model.phylo_model import PhyloModel


@pytest.mark.parametrize("calculate_clock_rate_gradient", [True, False])
def test_bito_direct_likelihood(
    topology_file, fasta_file, branch_lengths, model, calculate_clock_rate_gradient
):
    benchmarkable = BeagleDirectLikelihoodBenchmarkable()
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
