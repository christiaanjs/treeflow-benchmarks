from setuptools import setup

setup(
    name="treeflow-benchmarks",
    version="0.0.1",
    install_requires=["treeflow", "treeflow-paper", "snakemake", "click"],
    packages=["treeflow_benchmarks"],
    entry_points="""
        [console_scripts]
        treeflow_paper_benchmark=treeflow_benchmarks.cli:treeflow_paper_benchmark
        treeflow_paper_benchmark_error=treeflow_benchmarks.cli:treeflow_paper_benchmark_error
    """,
)
