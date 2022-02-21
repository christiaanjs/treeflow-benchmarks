from treeflow_benchmarks.treeflow_old import TreeflowOldLikelihoodBenchmarkable


def test_treeflow_old_likelihood(topology_file, fasta_file, branch_lengths, model):
    benchmarkable = TreeflowOldLikelihoodBenchmarkable()
    benchmarkable.initialize(topology_file, fasta_file, model)
    likelihood_res = benchmarkable.calculate_likelihoods(branch_lengths)
    assert likelihood_res.shape == (branch_lengths.shape[0],)
    gradient_res = benchmarkable.calculate_branch_gradients(branch_lengths)
    assert gradient_res.shape == branch_lengths.shape
