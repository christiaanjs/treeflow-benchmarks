from treeflow_benchmarks.jax import JaxLikelihoodBenchmarkable
import pytest


@pytest.mark.parametrize("jit", [False])
def test_jax_likelihood(topology_file, fasta_file, branch_lengths, model, jit):
    benchmarkable = JaxLikelihoodBenchmarkable(jit=jit)
    benchmarkable.initialize(topology_file, fasta_file, model)
    likelihood_res = benchmarkable.calculate_likelihoods(branch_lengths)
    assert likelihood_res.shape == (branch_lengths.shape[0],)
    gradient_res = benchmarkable.calculate_branch_gradients(branch_lengths)
    assert gradient_res.shape == branch_lengths.shape
