"""Compare likelihood values across all three methods (treeflow, bito/BEAGLE, JAX)."""
import yaml
import numpy as np
from treeflow.tree.io import parse_newick
from treeflow.model.phylo_model import PhyloModel
from treeflow_benchmarks.params import (
    get_numpy_gradient_params_dict,
    get_return_value_of_empty_generator,
)

DATA_DIR = "out/32-taxa/1-seed"
TOPOLOGY_FILE = f"{DATA_DIR}/topology-sim.newick"
FASTA_FILE = f"{DATA_DIR}/sequences.fasta"

MODEL_FILES = {
    "jc": "jc-model.yaml",
    "full (GTR+Weibull)": "full-model.yaml",
}


def check_model(model_name, model_file):
    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({model_file})")
    print(f"{'='*60}")

    with open(model_file) as f:
        model = yaml.safe_load(f)

    phylo_model = PhyloModel(model)
    params = get_numpy_gradient_params_dict(phylo_model, calculate_clock_rate_gradient=False)
    tree = parse_newick(TOPOLOGY_FILE)
    branch_lengths = tree.branch_lengths

    results = {}

    # 1. Treeflow (TensorFlow)
    print("\n--- Treeflow ---")
    try:
        from treeflow_benchmarks.treeflow import TreeflowLikelihoodBenchmarkable
        tf_bench = TreeflowLikelihoodBenchmarkable()
        tf_bench.initialize(TOPOLOGY_FILE, FASTA_FILE, model, False)
        ll = tf_bench.calculate_likelihoods(branch_lengths, params)
        results["treeflow"] = float(ll)
        print(f"  Log-likelihood: {ll}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 2. bito/BEAGLE
    #    beagle.py now applies clock_rate scaling internally.
    #    Note: bito uses different internal node ordering, so we call
    #    phylogenetic_likelihood directly and use bito's own branch lengths
    #    (copied before warm-up mutates them).
    print("\n--- bito/BEAGLE ---")
    try:
        from treeflow.acceleration.bito.beagle import phylogenetic_likelihood as bito_ll
        from treeflow.model.phylo_model import get_subst_model, get_subst_model_params
        subst_model = get_subst_model(phylo_model.subst_model)
        subst_params_bito, _ = get_return_value_of_empty_generator(
            get_subst_model_params(phylo_model.subst_model, phylo_model.subst_params)
        )
        log_prob_fn, inst = bito_ll(
            FASTA_FILE, subst_model,
            newick_file=TOPOLOGY_FILE, dated=True,
            clock_rate=phylo_model.clock_params["clock_rate"],
            site_model=phylo_model.site_model,
            site_model_params=phylo_model.site_params,
            **subst_params_bito,
        )
        # Copy branch lengths before any call mutates the internal tree
        bito_blens = np.array(inst.tree_collection.trees[0].branch_lengths)[:-1].copy()
        ll = log_prob_fn(bito_blens).numpy()
        results["bito"] = float(ll)
        print(f"  Log-likelihood: {ll}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 3. JAX (phylojax)
    print("\n--- JAX ---")
    try:
        from treeflow_benchmarks.jax import JaxLikelihoodBenchmarkable
        jax_bench = JaxLikelihoodBenchmarkable(jit=True)
        jax_bench.initialize(TOPOLOGY_FILE, FASTA_FILE, model, False)
        ll = jax_bench.calculate_likelihoods(branch_lengths, params)
        results["jax"] = float(ll)
        print(f"  Log-likelihood: {ll}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Compare
    if len(results) > 1:
        print("\n--- Comparison ---")
        methods = list(results.keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                diff = abs(results[methods[i]] - results[methods[j]])
                print(f"  |{methods[i]} - {methods[j]}| = {diff:.2e}")
        max_diff = max(
            abs(results[a] - results[b])
            for a in results for b in results if a != b
        )
        if max_diff < 1e-6:
            print(f"  PASS: All methods agree (max diff = {max_diff:.2e})")
        else:
            print(f"  WARNING: Max difference = {max_diff:.2e}")
    else:
        print(f"\nOnly {len(results)} method(s) succeeded, cannot compare.")


if __name__ == "__main__":
    for model_name, model_file in MODEL_FILES.items():
        check_model(model_name, model_file)
