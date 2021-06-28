import treeflow
import treeflow.tree_processing
import treeflow.tree_transform
import tensorflow as tf
import treeflow_benchmarks.benchmarking


def get_bij(topology_file, _class=treeflow.tree_transform.BranchBreaking, **kwargs):
    tree, taxon_names = treeflow.tree_processing.parse_newick(topology_file)
    topology = treeflow.tree_processing.update_topology_dict(tree["topology"])
    taxon_count = (tree["heights"].shape[0] + 1) // 2
    anchor_heights = treeflow.tree_processing.get_node_anchor_heights(
        tree["heights"], topology["postorder_node_indices"], topology["child_indices"]
    )
    anchor_heights = tf.convert_to_tensor(
        anchor_heights, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF
    )
    bij = _class(
        parent_indices=topology["parent_indices"][taxon_count:] - taxon_count,
        preorder_node_indices=topology["preorder_node_indices"][1:] - taxon_count,
        anchor_heights=anchor_heights,
        **kwargs
    )
    return bij, taxon_count


def get_ratios(topology_file, trees):
    bij, taxon_count = get_bij(topology_file)
    return tf.function(bij.inverse)(trees["heights"][..., taxon_count:])


class TensorflowRatioTransformBenchmarkable(
    treeflow_benchmarks.benchmarking.RatioTransformBenchmarkable
):
    def initialize(self, topology_file):
        self.bij, self.taxon_count = get_bij(topology_file)
        self.forward = tf.function(self.bij.forward)

        def grad(ratios, height_gradients):
            with tf.GradientTape() as t:
                t.watch(ratios)
                heights = self.bij.forward(ratios)
                return t.gradient(heights, ratios, output_gradients=height_gradients)

        self.grad = tf.function(grad)

        tree, _ = treeflow.tree_processing.parse_newick(topology_file)
        ratios = self.bij.inverse(tree["heights"][self.taxon_count :])
        self.forward(ratios)  # Do tracing
        self.grad(ratios, tf.ones_like(ratios))

    def calculate_heights(self, ratios):
        return self.forward(ratios)

    def calculate_ratio_gradients(self, ratios, height_gradients):
        return self.grad(ratios, height_gradients)
