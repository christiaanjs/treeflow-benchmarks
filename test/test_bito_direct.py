from treeflow_benchmarks.bito_direct import BeagleDirectLikelihoodBenchmarkable


def test_bito_direct_likelihood(topology_file, fasta_file, branch_lengths, model):
    benchmarkable = BeagleDirectLikelihoodBenchmarkable()
    benchmarkable.initialize(topology_file, fasta_file, model)
    likelihood_res = benchmarkable.calculate_likelihoods_loop(branch_lengths)
    assert likelihood_res.shape == (branch_lengths.shape[0],)
    gradient_res = benchmarkable.calculate_branch_gradients_loop(branch_lengths)
    assert gradient_res.shape == branch_lengths.shape
