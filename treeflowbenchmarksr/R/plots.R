

#' @export
comparisonPlot <- function(df, logScales=TRUE, scales="fixed", outFile=NULL, ggsaveArgs=list()){
  plotDf <- df %>%
    dplyr::mutate(computation=as.factor(computation), model=as.factor(model))
  plot <- ggplot2::ggplot(plotDf, ggplot2::aes(x=taxon_count, y=time, group=interaction(taxon_count, computation, model), colour=interaction(computation, model))) +
    ggplot2::geom_boxplot() +
    if(scales == "fixed"){ ggplot2::facet_grid(~ method, scales=scales) } else { ggplot2::facet_wrap(~ method, scales=scales) }# + ggplot2::theme(legend.position = "bottom")
  finalPlot <- if(logScales){
    plot + ggplot2::scale_y_log10() + ggplot2::scale_x_continuous(trans="log2") + ggplot2::geom_abline(intercept=0, slope=1)
  } else {
    plot
  }
  if(!is.null(outFile)){
    do.call(ggplot2::ggsave, modifyList(ggsaveArgs, list(filename=outFile, plot=finalPlot)))
  }
  finalPlot
}
