

#' @export
comparisonPlot <- function(df, logScales = TRUE, scales = "fixed", outFile = NULL, ggsaveArgs = list(), colorFunc = interaction, renameFunc = identity) {
  plotDf <- df %>%
    dplyr::mutate(
      computation = factor(computation, levels = unique(computation)),
      model = factor(model, levels = unique(model)),
      method = factor(method, levels = unique(method)),
    ) %>%
    dplyr::mutate(
      group = interaction(taxon_count, computation, model),
      colour = colorFunc(computation, model)
    ) %>%
    dplyr::mutate(colour = factor(colour, levels = unique(colour))) %>%
    dplyr::rename_with(renameFunc)
  plot <- ggplot2::ggplot(
    plotDf,
    ggplot2::aes_(
      x = as.name(renameFunc("taxon_count")),
      y = as.name(renameFunc("time")),
      group = ~group,
      colour = as.name(renameFunc("colour"))
    )
  ) +
    ggplot2::geom_boxplot(position = "identity") +
    if (scales == "fixed") {
      ggplot2::facet_grid(reformulate(renameFunc("method")), scales = scales)
    } else {
      ggplot2::facet_wrap(reformulate(renameFunc("method")), scales = scales)
    } # + ggplot2::theme(legend.position = "bottom")
  finalPlot <- if (logScales) {
    plot + ggplot2::scale_y_log10() + ggplot2::scale_x_continuous(trans = "log2") + ggplot2::geom_abline(intercept = 0, slope = 1)
  } else {
    plot
  }
  if (!is.null(outFile)) {
    do.call(ggplot2::ggsave, modifyList(ggsaveArgs, list(filename = outFile, plot = finalPlot)))
  }
  finalPlot
}
