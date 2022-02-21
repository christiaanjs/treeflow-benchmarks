import pytest
from treeflow_benchmarks.simulation import simulate_heights


@pytest.fixture
def newick_out():
    return "/tmp/height-sim.newick"


def test_simulate_heights(topology_file, newick_out):
    simulate_heights(
        dict(sample_count=3, height_scale=0.1), topology_file, 2, newick_out
    )
