import pathlib

from treeflow_pipeline.util import text_output, yaml_input, yaml_output, pickle_output
import treeflow_pipeline.simulation as pipe_sim
import treeflow_pipeline.templating as tem
import treeflow_pipeline.model as mod
import treeflow_benchmarks.simulation as bench_sim
import treeflow_pipeline.topology_inference as top


configfile: "config.yaml"

wd = pathlib.Path(config["working_directory"])
taxon_dir = "{taxon_count}taxa"

rule benchmarks:
    input:
        expand(str(wd / taxon_dir / "tree-sim.newick"), taxon_count=config["taxon_counts"])

rule model_params:
    input:
        model = config["model_file"]
    output:
        params = wd / "params.yaml"
    run:
        yaml_output(mod.Model(yaml_input(input.model)).free_params(), output.params)


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
        pathlib.Path(rules.topology_sim.output.trees).with_suffix(".newick")
    group: "sim"
    run:
        top.convert_tree(input[0], 'nexus', output[0], 'newick')

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
