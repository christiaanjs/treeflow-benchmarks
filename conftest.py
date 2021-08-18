import pytest
import pathlib
import treeflow.sequences
import pickle
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
    with open(test_data_dir / "tree-sim.pickle", "rb") as f:
        tree = pickle.load(f)

    return treeflow.sequences.get_branch_lengths(tree).numpy()


@pytest.fixture
def model(test_data_dir):
    return mod.Model(yaml_input(test_data_dir / "model.yaml"))
