# treeflow-benchmarks

Benchmarking pipeline for phylogenetic likelihood computation, comparing [TreeFlow](https://github.com/christiaanjs/treeflow), [BEAGLE](https://github.com/beagle-dev/beagle-lib)/[bito](https://github.com/phylovi/bito), and [JAX](https://github.com/christiaanjs/phylojax) implementations. Produces the benchmark figure (Fig 7) in the [TreeFlow paper](https://github.com/christiaanjs/treeflow-paper).

## What it does

1. **Simulates** coalescent tree topologies and DNA sequences using BEAST 2
2. **Benchmarks** likelihood and gradient computation across methods, taxon counts (32–2048), and models (JC, full GTR+Gamma)
3. **Aggregates** timing results across 10 replicate simulations per configuration
4. **Generates** log-scale comparison plots and fits log-log scaling exponents

## Requirements

### Python

- Python >= 3.9
- [`treeflow`](https://github.com/christiaanjs/treeflow) and its dependencies
- [`treeflow-paper`](https://github.com/christiaanjs/treeflow-paper) (provides the `treeflow_pipeline` package)
- [`phylojax`](https://github.com/christiaanjs/phylojax) (JAX-based likelihood)
- [`bito`](https://github.com/phylovi/bito) (BEAGLE integration; must be built from source)

### External tools

- [BEAST 2](https://www.beast2.org/) >= 2.7, with the [Feast](https://github.com/tgvaughan/feast) package installed
- [BEAGLE](https://github.com/beagle-dev/beagle-lib) (required for the BEAGLE/bito benchmarks)

### R

- R >= 4.0 with packages: `ggplot2`, `dplyr`, `tidyr`, `broom`, `purrr`, `readr`
- The `treeflowbenchmarksr` R package (included in this repo):

```bash
R -e 'install.packages("treeflowbenchmarksr", repos = NULL, type = "source")'
```

## Installation

```bash
pip install -e .
pip install -e /path/to/treeflow
pip install -e /path/to/treeflow-paper
pip install -e /path/to/phylojax
```

For bito, follow the [build instructions](https://github.com/phylovi/bito#readme) in its repository.

Ensure `beast` is available on your `PATH`.

## Running

```bash
snakemake --cores 1
```

This runs the full pipeline (~1000 steps). Individual targets can also be requested:

```bash
snakemake out/plot-data.csv --cores 1    # timing data only
snakemake out/log-scale-plot.png --cores 1  # plot only (requires timing data)
```

## Configuration

`config.yaml` controls the pipeline parameters:

- `full_taxon_counts` — taxon counts to benchmark (default: 32, 64, 128, 256, 512, 1024, 2048)
- `replicates` — number of replicate simulations per configuration (default: 10)
- `sequence_length` — simulated alignment length (default: 1000)
- `full_benchmarkables` — methods to benchmark (default: treeflow, beagle\_bito\_direct)
- `short_benchmarkables` — methods only run at smaller taxon counts (default: jax)
- `tasks` — computations to time (default: likelihood)

Model specifications are in `sim-model.yaml` (simulation), `jc-model.yaml` (JC inference), and `full-model.yaml` (GTR+Gamma inference).

## Outputs

- `out/plot-data.csv` — timing data in long format (method, taxon\_count, seed, model, computation, time)
- `out/fit-table.csv` — log-log scaling exponents per method
- `out/log-scale-plot.png` — benchmark comparison plot (consumed by the treeflow-paper manuscript pipeline)