from importlib.resources import path
import pathlib
import tempfile
from treeflow.tree.io import parse_newick
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    convert_tree_to_tensor,
    TensorflowRootedTree,
)
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow_pipeline.util import pickle_input, pickle_output


def test_tensorflow_tree_pickling(topology_file):
    tree = convert_tree_to_tensor(parse_newick(topology_file))
    pickle_file_path = pathlib.Path(tempfile.gettempdir()) / "tree.pickle"
    pickle_output(tree, pickle_file_path)
    unpickled = pickle_input(pickle_file_path)
    assert isinstance(unpickled, TensorflowRootedTree)
    numpy_tree = unpickled.numpy()
    assert tuple(numpy_tree.taxon_set) == tuple(tree.taxon_set)


def test_numpy_tree_pickling(topology_file):
    tree = parse_newick(topology_file)
    pickle_file_path = pathlib.Path(tempfile.gettempdir()) / "tree.pickle"
    pickle_output(tree, pickle_file_path)
    unpickled = pickle_input(pickle_file_path)
    assert isinstance(unpickled, NumpyRootedTree)
    assert tuple(unpickled.taxon_set) == tuple(tree.taxon_set)
