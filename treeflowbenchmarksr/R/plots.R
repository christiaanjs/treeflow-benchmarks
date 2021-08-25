

#' @export
comparisonPlot <- function(plotData, logScales=TRUE, scales="fixed", outFile=NULL, ggsaveArgs=list()){
  df <- readr::read_csv(plotData, show_col_types = FALSE)
  plotDf <- df %>%
    dplyr::mutate(computation=as.factor(computation))
  plot <- ggplot2::ggplot(plotDf, ggplot2::aes(x=taxon_count, y=time, group=interaction(taxon_count, computation), colour=computation)) +
    ggplot2::geom_boxplot()+
    ggplot2::facet_wrap(~ method, scales=scales)
  finalPlot <- if(logScales){
    plot + ggplot2::scale_y_log10() + ggplot2::scale_x_continuous(trans="log2") + ggplot2::geom_abline(intercept=0, slope=1)
  } else {
    plot
  }
  if(!is.null(outFile)){
    do.call(ggplot2::ggsave, modifyList(ggsaveArgs, list(filename=outFile, plot=finalPlot)))
  }
  plot
}
