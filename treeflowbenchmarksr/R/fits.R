#' @export
fitLogLogLine <- function(plotData){
  df <- readr::read_csv(plotData)
  fitParameters <- df %>%
    tidyr::nest(data=c(seed, taxon_count, time)) %>%
    dplyr::mutate(
      fit=purrr::map(data, ~ lm(log(time) ~ log(taxon_count), data=.))
    ) %>%
    dplyr::mutate(tidied=purrr::map(fit, broom::tidy)) %>%
    tidyr::unnest(tidied)
  fitParameters %>%
    tidyr::pivot_wider(
      id_cols=c(method, computation),
      names_from = term,
      values_from=estimate
    ) %>%
    dplyr::rename(slope=`log(taxon_count)`, intercept=`(Intercept)`)
}