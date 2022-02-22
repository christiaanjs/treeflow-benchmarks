import pytest
from treeflow_benchmarks.simulation import simulate_heights
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree


@pytest.fixture
def newick_out():
    return "/tmp/height-sim.newick"


def test_simulate_heights(topology_file, newick_out):
    res = simulate_heights(
        dict(sample_count=3, height_scale=0.1), topology_file, 2, newick_out
    )
    assert isinstance(res, NumpyRootedTree)
    branch_lengths = res.branch_lengths
