import tensorflow as tf
import tensorflow_probability as tfp
import treeflow.model
import treeflow.sequences
import treeflow_pipeline.results


def simulate_heights(config, topology_file, seed, newick_out):
    tree_approx, vars = treeflow.model.construct_tree_approximation(topology_file)
    scale_var = vars["scale_inverse_softplus"]
    scale_var.assign(
        tfp.math.softplus_inverse(tf.ones_like(scale_var) * config["height_scale"])
    )
    samples = tree_approx.sample(config["sample_count"], seed=seed)
    treeflow_pipeline.results.write_tensor_trees(
        topology_file, treeflow.sequences.get_branch_lengths(samples), newick_out
    )
    return samples
