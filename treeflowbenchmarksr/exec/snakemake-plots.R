ggsaveArgs <- list(width=8, height=4)
df <- readr::read_csv(snakemake@input[["plot_data"]], show_col_types = FALSE)
treeflowbenchmarksr::comparisonPlot(
    df,
    outFile=snakemake@output[["log_scale_plot"]],
    ggsaveArgs=ggsaveArgs
)
treeflowbenchmarksr::comparisonPlot(
    df,
    snakemake@input[["plot_data"]],
    logScales=FALSE,
    scales="free_y",
    outFile=snakemake@output[["free_scale_plot"]],
    ggsaveArgs=ggsaveArgs
)
