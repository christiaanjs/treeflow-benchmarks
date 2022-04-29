import click
import numpy as np

from treeflow_benchmarks.benchmarkables import (
    benchmarkables as benchables,
    benchmark_functions,
    output_types,
)

from treeflow_pipeline.util import (
    text_input,
    text_output,
    yaml_input,
    yaml_output,
    pickle_input,
    pickle_output,
)
import treeflow_benchmarks.benchmarking as bench


@click.command()
@click.option("--task", type=click.Choice(benchmark_functions.keys()), required=True)
@click.option("--benchmarkable", required=True)
@click.option("--model-file", type=click.Path(exists=True), required=True)
@click.option("--output-file", type=click.Path(), required=True)
@click.option("--calculate-clock-rate-gradient", is_flag=True)
@click.option("--sim-state", type=click.Path(exists=True), required=True)
@click.option("--taxon-count", type=int, required=True)
@click.option("--seed", type=int, required=True)
@click.option("--model-name", required=True)
def treeflow_paper_benchmark(
    task,
    benchmarkable,
    model_file,
    output_file,
    calculate_clock_rate_gradient,
    sim_state,
    taxon_count,
    seed,
    model_name,
):
    pickle_output(
        bench.annotate_times(
            benchmark_functions[task](
                benchmarkable=benchables[task][benchmarkable],
                model=yaml_input(model_file),
                calculate_clock_rate_gradient=calculate_clock_rate_gradient,
                **pickle_input(sim_state)
            ),
            taxon_count=taxon_count,
            seed=seed,
            method=benchmarkable,
            model=model_name,
        ),
        output_file,
    )


@click.command()
@click.option("--task", type=click.Choice(benchmark_functions.keys()), required=True)
@click.option("--benchmarkable", required=True)
@click.option("--output-file", type=click.Path(), required=True)
@click.option("--taxon-count", type=int, required=True)
@click.option("--seed", type=int, required=True)
@click.option("--model-name", required=True)
def treeflow_paper_benchmark_error(
    task,
    benchmarkable,
    output_file,
    taxon_count,
    seed,
    model_name,
):
    output_type = output_types[task]
    pickle_output(
        bench.annotate_times(
            output_type(**{field: np.nan for field in output_type._fields}),
            taxon_count=taxon_count,
            seed=seed,
            method=benchmarkable,
            model=model_name,
        ),
        output_file,
    )
