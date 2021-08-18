import pathlib

from treeflow_pipeline.util import text_input, text_output, yaml_input, yaml_output, pickle_input, pickle_output
import treeflow_pipeline.simulation as pipe_sim
import treeflow_pipeline.templating as tem
import treeflow_pipeline.model as mod
import treeflow_benchmarks.simulation as bench_sim
import treeflow_pipeline.topology_inference as top
import treeflow_benchmarks.benchmarking as bench
import treeflow_benchmarks.tf_likelihood as bench_tf
import treeflow_benchmarks.libsbn as bench_libsbn
import treeflow_benchmarks.tree_transform as bench_trans
import treeflow_benchmarks.benchmarkables as benchmarkables
import pandas as pd

config_file = "config.yaml"
configfile: config_file

wd = pathlib.Path(config["working_directory"])
taxon_dir = "{taxon_count}taxa"
seed_dir = "{seed}seed"
likelihood_dir = "{likelihood}"
ratio_dir = "ratio_{ratio}"

seeds = [str(i+1) for i in range(config["replicates"])]

def get_model(model_file):
    return mod.Model(yaml_input(model_file))

rule benchmarks:
    input:
        wd / "likelihood-times.csv",
        # wd / "plot-benchmarks.html",
        # wd / "log-scale-plot.png",
        # wd / "free-scale-plot.png",
        # wd / "rmd-report.html"


rule model_params:
    input:
        model = config["model_file"]
    output:
        params = wd / "params.yaml"
    run:
        yaml_output(get_model(input.model).free_params(), output.params)


rule sampling_times:
    output:
        times = wd / taxon_dir / "sampling-times.yaml"
    run:
        yaml_output(pipe_sim.get_sampling_times(config, int(wildcards.taxon_count)), output.times)

trees_filename = "topology-sim.trees"
rule topology_sim_xml:
    input:
        sampling_times = rules.sampling_times.output.times,
        params = rules.model_params.output.params
    output:
        xml = wd / taxon_dir / seed_dir / "topology-sim.xml" # Use seed_dir to set output filename
    group: "sim"
    run:
        text_output(
            tem.build_tree_sim(
                config,
                yaml_input(input.sampling_times),
                yaml_input(input.params),
                output.xml,
                trees_filename=trees_filename
            ),
            output.xml
        )

rule topology_sim:
    input:
        rules.topology_sim_xml.output.xml
    output:
        trees = wd / taxon_dir / seed_dir / trees_filename
    group: "sim"
    shell:
        "beast -seed {wildcards.seed} {input}"

rule topology_sim_newick:
    input:
        rules.topology_sim.output.trees
    output:
        newick = pathlib.Path(rules.topology_sim.output.trees).with_suffix(".newick")
    group: "sim"
    run:
        top.convert_tree(input[0], 'nexus', output.newick, 'newick')

def sibling_file(path, filename):
    return pathlib.Path(path).parents[0] / filename

rule height_sim:
    input:
        topology_file = rules.topology_sim_newick.output.newick
    output:
        pickle = sibling_file(rules.topology_sim_newick.output.newick, "tree-sim.pickle"),
        newick = sibling_file(rules.topology_sim_newick.output.newick, "tree-sim.newick")
    run:
        pickle_output(
            bench_sim.simulate_heights(config, input.topology_file, int(wildcards.seed), output.newick),
            output.pickle
        )


rule sequence_sim_xml:
    input:
        model = config["model_file"],
        topology = rules.topology_sim_newick.output.newick,
        params = rules.model_params.output.params,
        sampling_times = rules.sampling_times.output.times,
    params:
        model = lambda wildcards, input: get_model(input.model)
    output:
        xml = sibling_file(rules.topology_sim_newick.output.newick, "sequence-sim.xml")
    run:
        text_output(
            tem.build_sequence_sim(
                dict(config, **params.model.subst_params),
                params.model,
                text_input(input.topology),
                yaml_input(input.params),
                None,
                yaml_input(input.sampling_times),
                config["sequence_length"],
                output.xml
            ),
            output.xml
        )

rule sequence_sim:
    input:
        xml = rules.sequence_sim_xml.output.xml
    output:
        sequences = sibling_file(rules.sequence_sim_xml.output.xml, "sequences.xml")
    shell:
        "beast -seed {wildcards.seed} {input}"

rule fasta_sim:
    input:
        xml = rules.sequence_sim.output.sequences
    output:
        fasta = sibling_file(rules.sequence_sim.output.sequences, "sequences.fasta")
    run:
        pipe_sim.convert_simulated_sequences(input.xml, output.fasta, 'fasta')

target_benchmarkables = set(["tensorflow", "jax", "jax_jit"])
likelihood_benchmarkables = { 
    key: value for key, value in benchmarkables.likelihood_benchmarkables.items() if key in target_benchmarkables
}

rule likelihood_times:
    input:
        topology = rules.topology_sim_newick.output.newick,
        heights = rules.height_sim.output.pickle,
        sequences = rules.fasta_sim.output.fasta,
        model = config["model_file"],
    output:
        times = wd / taxon_dir / seed_dir / likelihood_dir / "likelihood-times.pickle"
    params:
        model = lambda wildcards, input: get_model(input.model)
    run:
        pickle_output(
            bench.annotate_times(
                bench.benchmark_likelihood(
                    input.topology,
                    input.sequences,
                    params.model,
                    pickle_input(input.heights),
                    likelihood_benchmarkables[wildcards.likelihood]
                ),
                taxon_count=wildcards.taxon_count,
                seed=wildcards.seed,
                method=wildcards.likelihood
            ), 
            output.times
        )

rule likelihood_times_csv:
    input:
        times = expand(rules.likelihood_times.output.times,
            taxon_count=config["taxon_counts"],
            seed=seeds,
            likelihood=likelihood_benchmarkables.keys()
        )
    output:
        csv = wd / "likelihood-times.csv"
    run:
        pd.DataFrame([pickle_input(x) for x in input.times]).to_csv(output.csv, index=False)

rule ratios:
    input:
        topology = rules.topology_sim_newick.output.newick,
        heights = rules.height_sim.output.pickle,
    output:
        ratios = sibling_file(rules.height_sim.output.pickle, "ratios.pickle")
    run:
        pickle_output(bench_trans.get_ratios(input.topology, pickle_input(input.heights)), output.ratios)

ratio_transform_benchmarkables = benchmarkables.ratio_transform_benchmarkables

rule transform_times:
    input:
        topology = rules.topology_sim_newick.output.newick,
        ratios = rules.ratios.output.ratios,
        trees = rules.height_sim.output.pickle,
    output:
        times = wd / taxon_dir / seed_dir / ratio_dir / "ratio-times.pickle"
    run:
        pickle_output(
            bench.annotate_times(
                bench.benchmark_ratio_transform(
                    input.topology,
                    pickle_input(input.ratios),
                    pickle_input(input.trees),
                    ratio_transform_benchmarkables[wildcards.ratio]
                ),
                taxon_count=wildcards.taxon_count,
                seed=wildcards.seed,
                method=wildcards.ratio
            ), 
            output.times
        )


rule transform_times_csv:
    input:
        times = expand(rules.transform_times.output.times,
            taxon_count=config["taxon_counts"],
            seed=seeds,
            ratio=ratio_transform_benchmarkables.keys()
        )
    output:
        csv = wd / "transform-times.csv"
    run:
        pd.DataFrame([pickle_input(x) for x in input.times]).to_csv(output.csv, index=False)

rule report_notebook:
    input:
        notebook = "notebook/plot-benchmarks.ipynb",
        likelihood_times = rules.likelihood_times_csv.output.csv,
        transform_times = rules.transform_times_csv.output.csv,
        config = config_file,
        model = config["model_file"],
    output:
        notebook = wd / "plot-benchmarks.ipynb",
        plot_data = wd / "plot-data.csv"
    shell:
        """
        papermill {input.notebook} {output.notebook} \
            -p likelihood_times_file {input.likelihood_times} \
            -p transform_times_file {input.transform_times} \
            -p config_file {input.config} \
            -p model_file {input.model} \
            -p plot_data_output {output.plot_data}
        """

rule report:
    input:
        notebook = rules.report_notebook.output.notebook
    output:
        html = wd / "plot-benchmarks.html"
    shell:
        "jupyter nbconvert --to html {input.notebook}"

rule plots:
    input:
        plot_data = rules.report_notebook.output.plot_data
    output:
        log_scale_plot = wd / "log-scale-plot.png",
        free_scale_plot = wd / "free-scale-plot.png"
    script:
        "treeflowbenchmarksr/exec/snakemake-plots.R"

rule rmd_report:
    input:
        plot_data = rules.report_notebook.output.plot_data
    output:
        rmd_report = wd / "rmd-report.html"
    script:
        "treeflowbenchmarksr/exec/report.Rmd"