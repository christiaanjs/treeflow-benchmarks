#' @export
comparisonPlot <- function(plotData, logScales=TRUE, scales="fixed"){
  df <- readr::read_csv(plotData)
  plotDf <- df %>%
    dplyr::mutate(computation=as.factor(computation))
  plot <- ggplot2::ggplot(plotDf, ggplot2::aes(x=taxon_count, y=time, group=interaction(taxon_count, computation), colour=computation)) +
    ggplot2::geom_boxplot()+
    ggplot2::facet_wrap(~ method, scales=scales)
  if(logScales){
    plot + ggplot2::scale_y_log10() + ggplot2::scale_x_continuous(trans="log2")
  } else {
    plot
  }

}
