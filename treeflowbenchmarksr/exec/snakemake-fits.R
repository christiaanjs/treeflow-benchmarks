df <- readr::read_csv(snakemake@input[["plot_data"]], show_col_types = FALSE)
fitTable <- treeflowbenchmarksr::fitLogLogLine(df)
readr::write_csv(fitTable, snakemake@output[["fit_table"]])
