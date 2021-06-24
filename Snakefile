import pathlib

from treeflow_pipeline.util import text_input, text_output, yaml_input, yaml_output, pickle_output
import treeflow_pipeline.simulation as pipe_sim
import treeflow_pipeline.templating as tem
import treeflow_pipeline.model as mod
import treeflow_benchmarks.simulation as bench_sim
import treeflow_pipeline.topology_inference as top


configfile: "config.yaml"

wd = pathlib.Path(config["working_directory"])
taxon_dir = "{taxon_count}taxa"

def get_model(model_file):
    return mod.Model(yaml_input(model_file))

rule benchmarks:
    input:
        expand(str(wd / taxon_dir / "sequences.fasta"), taxon_count=config["taxon_counts"])

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
        xml = wd / taxon_dir / "topology-sim.xml"
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
        trees = wd / taxon_dir  / trees_filename
    group: "sim"
    shell:
        "beast -seed {config[seed]} {input}"

rule topology_sim_newick:
    input:
        rules.topology_sim.output.trees
    output:
        newick = pathlib.Path(rules.topology_sim.output.trees).with_suffix(".newick")
    group: "sim"
    run:
        top.convert_tree(input[0], 'nexus', output.newick, 'newick')

rule height_sim:
    input:
        topology_file = wd / taxon_dir  / "topology-sim.newick"
    output:
        pickle = wd / taxon_dir / "tree-sim.pickle",
        newick = wd / taxon_dir / "tree-sim.newick"
    run:
        pickle_output(
            bench_sim.simulate_heights(config, input.topology_file, config["seed"], output.newick),
            output.pickle
        )


rule sequence_sim_xml:
    input:
        tree_file = rules.height_sim.output.newick,
        model = config["model_file"],
        topology = rules.topology_sim_newick.output.newick,
        params = rules.model_params.output.params,
        sampling_times = rules.sampling_times.output.times,
    params:
        model = lambda wildcards, input: get_model(input.model)
    output:
        xml = wd / taxon_dir / "sequence-sim.xml"
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
        sequences = pathlib.Path(rules.sequence_sim_xml.output.xml).parents[0] / "sequences.xml"
    shell:
        "beast -seed {config[seed]} {input}"

rule fasta_sim:
    input:
        xml = rules.sequence_sim.output.sequences
    output:
        fasta = wd / taxon_dir / "sequences.fasta"
    run:
        pipe_sim.convert_simulated_sequences(input.xml, output.fasta, 'fasta')


rule tf_likelihood_times:
    input:
        heights = rules.height_sim.output.pickle,
        sequences = rules.fasta_sim.output