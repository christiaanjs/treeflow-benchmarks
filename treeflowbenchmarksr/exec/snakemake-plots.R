ggsaveArgs <- list(width=8, height=4)
treeflowbenchmarksr::comparisonPlot(
    snakemake@input[["plot_data"]],
    outFile=snakemake@output[["log_scale_plot"]],
    ggsaveArgs=ggsaveArgs
)
treeflowbenchmarksr::comparisonPlot(
    snakemake@input[["plot_data"]],
    logScales=FALSE,
    scales="free_y",
    outFile=snakemake@output[["free_scale_plot"]],
    ggsaveArgs=ggsaveArgs
)
