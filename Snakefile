import pathlib

from treeflow_pipeline.util import text_input, text_output, yaml_input, yaml_output, pickle_input, pickle_output
import treeflow_pipeline.simulation as pipe_sim
import treeflow_pipeline.templating as tem
import treeflow_pipeline.model as mod
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
benchmarkable_wildcard = "{benchmarkable}"

seeds = [str(i+1) for i in range(config["replicates"])]

all_computations = [time_field for task in config["tasks"] for time_field in output_types[task]._fields]
assert len(all_computations) == len(set(all_computations)), "Time field names must be unique"

def get_model(model_file):
    return mod.Model(yaml_input(model_file))

rule benchmarks:
    input:
        wd / "rmd-report.html",
        wd / "log-scale-plot.png"


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
        model = config["model_file"],
    output:
        sim_state = sibling_file(rules.height_sim.output.pickle, "sim-state.pickle")
    run:
        pickle_output(dict(
            topology_file=input.topology_file,
            trees=pickle_input(input.trees),
            fasta_file=input.fasta_file,
            ratios=pickle_input(input.ratios),
            model=get_model(input.model)
        ), output.sim_state)

rule times:
    input:
        sim_state = rules.sim_state.output.sim_state
    output:
        times = wd / taxon_dir / seed_dir / task_dir / (benchmarkable_wildcard + "-times.csv")
    run:
        pickle_output(
            bench.annotate_times(
                benchmark_functions[wildcards.task](
                    benchmarkable=benchables[wildcards.task][wildcards.benchmarkable],
                    **pickle_input(input.sim_state)
                ),
                taxon_count=wildcards.taxon_count,
                seed=wildcards.seed,
                method=wildcards.benchmarkable
            ),
            output.times
        )

rule task_times_csv:
    input:
        times = expand(
            rules.times.output.times,
            taxon_count=config["taxon_counts"],
            seed=seeds,
            benchmarkable=config["benchmarkables"],
            allow_missing=True
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
                id_vars=["method", "seed", "taxon_count"],
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

rule rmd_report:
    input:
        plot_data = rules.plot_data.output.csv
    output:
        rmd_report = wd / "rmd-report.html"
    script:
        "treeflowbenchmarksr/exec/report.Rmd"

