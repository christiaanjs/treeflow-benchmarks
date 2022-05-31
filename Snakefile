import pathlib

from treeflow_pipeline.util import text_input, text_output, yaml_input, yaml_output, pickle_input, pickle_output
import treeflow_pipeline.simulation as pipe_sim
import treeflow_pipeline.templating as tem
from treeflow.model.phylo_model import PhyloModel
import treeflow_benchmarks.simulation as bench_sim
import treeflow_pipeline.topology_inference as top
import treeflow_benchmarks.benchmarking as bench
from treeflow_benchmarks.benchmarkables import benchmarkables as benchables, benchmark_functions, output_types
import treeflow_benchmarks.tree_transform as bench_trans
import pandas as pd
from functools import reduce

config_file = "config.yaml"
configfile: config_file

wd = pathlib.Path(config["working_directory"])
taxon_dir = "{taxon_count}-taxa"
seed_dir = "{seed}-seed"
task_dir = "{task}-task"
model_dir = "{model}-model"
benchmarkable_wildcard = "{benchmarkable}"

seeds = [str(i+1) for i in range(config["replicates"])]

all_computations = [time_field for task in config["tasks"] for time_field in output_types[task]._fields]
assert len(all_computations) == len(set(all_computations)), "Time field names must be unique"

def get_model(model_file):
    return PhyloModel(yaml_input(model_file))

model_files_and_clock_rate_grad = dict(
    jc=("jc-model.yaml", False),
    full=("full-model.yaml", True)
)
model_files = { key: value[0] for key, value in model_files_and_clock_rate_grad.items() }
calculate_clock_rate_grad = { key: value[1] for key, value in model_files_and_clock_rate_grad.items() }
sim_model = "sim-model.yaml"

rule benchmarks:
    input:
        wd / "rmd-report.pdf",
        wd / "log-scale-plot.png",
        wd / "fit-table.csv"


rule model_params:
    input:
        model = sim_model
    output:
        params = wd / "params.yaml"
    run:
        yaml_output(get_model(input.model).all_params(), output.params)


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
        model = sim_model,
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

rule ratios:
    input:
        topology = rules.topology_sim_newick.output.newick,
        heights = rules.height_sim.output.pickle,
    output:
        ratios = sibling_file(rules.height_sim.output.pickle, "ratios.pickle")
    run:
        pickle_output(bench_trans.get_ratios(input.topology, pickle_input(input.heights)), output.ratios)

rule sim_state:
    input:
        topology_file = rules.topology_sim_newick.output.newick,
        trees = rules.height_sim.output.pickle,
        fasta_file = rules.fasta_sim.output.fasta,
        ratios = rules.ratios.output.ratios,
    output:
        sim_state = sibling_file(rules.height_sim.output.pickle, "sim-state.pickle")
    run:
        pickle_output(dict(
            topology_file=input.topology_file,
            trees=pickle_input(input.trees),
            fasta_file=input.fasta_file,
            ratios=pickle_input(input.ratios),
        ), output.sim_state)

rule times:
    input:
        sim_state = rules.sim_state.output.sim_state
    output: # TODO: Change to pickle
        times = wd / taxon_dir / seed_dir / model_dir / task_dir / (benchmarkable_wildcard + "-times.csv")
    params:
        model_file = lambda wildcards: model_files[wildcards.model],
        clock_rate_flag = lambda wildcards: "--calculate-clock-rate-gradient" if calculate_clock_rate_grad[wildcards.model] else ""
    shell:
        """
        treeflow_paper_benchmark \
            --task {wildcards.task} \
            --benchmarkable {wildcards.benchmarkable} \
            --model-file {params.model_file} \
            --output-file {output.times} \
            {params.clock_rate_flag} \
            --sim-state {input.sim_state} \
            --taxon-count {wildcards.taxon_count} \
            --seed {wildcards.seed} \
            --model-name {wildcards.model} || \
        treeflow_paper_benchmark_error \
            --task {wildcards.task} \
            --benchmarkable {wildcards.benchmarkable} \
            --output-file {output.times} \
            --taxon-count {wildcards.taxon_count} \
            --seed {wildcards.seed} \
            --model-name {wildcards.model}
        """

rule task_times_csv:
    input:
        times = (
            list(expand(
                rules.times.output.times,
                taxon_count=config["full_taxon_counts"],
                seed=seeds,
                benchmarkable=config["full_benchmarkables"],
                model=list(model_files.keys()),
                allow_missing=True
            )) +
            list(expand(
                rules.times.output.times,
                taxon_count=config["short_taxon_counts"],
                seed=seeds,
                benchmarkable=config["short_benchmarkables"],
                model=list(model_files.keys()),
                allow_missing=True
            ))
        )
    output:
        csv = wd /  (task_dir + "-times.csv")
    run:
        pd.DataFrame([pickle_input(x) for x in input.times]).to_csv(output.csv, index=False)


rule times_csv:
    input:
        csvs = expand(
            rules.task_times_csv.output.csv, task=config["tasks"]
        )
    output:
        csv = wd / "times.csv"
    run:
        reduce(pd.merge, [pd.read_csv(x) for x in input.csvs]).to_csv(output.csv, index=False)

rule plot_data:
    input:
        times_csv = rules.times_csv.output.csv
    output:
        csv = wd / "plot-data.csv"
    run:
        (pd.read_csv(input.times_csv)
            .melt(
                id_vars=["method", "seed", "taxon_count", "model"],
                var_name="computation",
                value_name="time"
            )
        ).to_csv(output.csv, index=False)


rule plots:
    input:
        plot_data = rules.plot_data.output.csv
    output:
        log_scale_plot = wd / "log-scale-plot.png",
        free_scale_plot = wd / "free-scale-plot.png"
    script:
        "treeflowbenchmarksr/exec/snakemake-plots.R"

rule fit_table:
    input:
        plot_data = rules.plot_data.output.csv
    output:
        fit_table = wd / "fit-table.csv"
    script:
        "treeflowbenchmarksr/exec/snakemake-fits.R"

rule rmd_report:
    input:
        plot_data = rules.plot_data.output.csv
    output:
        rmd_report = wd / "rmd-report.pdf"
    script:
        "treeflowbenchmarksr/exec/report.Rmd"

