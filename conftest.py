import pytest
import pathlib
import pickle
import numpy as np
import dendropy
import treeflow_pipeline.model as mod
from treeflow_pipeline.util import yaml_input


@pytest.fixture
def test_data_dir():
    return pathlib.Path("test") / "data"


@pytest.fixture
def topology_file(test_data_dir):
    return str(test_data_dir / "topology-sim.newick")


@pytest.fixture
def fasta_file(test_data_dir):
    return str(test_data_dir / "sequences.fasta")


@pytest.fixture
def branch_lengths(test_data_dir):
    tree_list = dendropy.TreeList.get(
        path=str(test_data_dir / "height-sim.nexus"), schema="nexus"
    )
    branch_lengths = np.array(
        [
            [edge.length for edge in tree.postorder_edge_iter()][:-1]
            for tree in tree_list
        ]
    )
    return branch_lengths


@pytest.fixture(params=["model.yaml", "jc_model.yaml"])
def model(test_data_dir, request):
    return yaml_input(test_data_dir / request.param)
