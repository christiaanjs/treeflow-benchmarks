#' @export
fitLogLogLine <- function(df){
  fitParameters <- df %>%
    dplyr::select(method, computation, model, seed, taxon_count, time) %>%
    tidyr::nest(data=c(seed, taxon_count, time)) %>%
    dplyr::mutate(
      fit=purrr::map(data, ~ lm(log(time) ~ log(taxon_count), data=.))
    ) %>%
    dplyr::mutate(tidied=purrr::map(fit, broom::tidy)) %>%
    tidyr::unnest(tidied)
  fitParameters %>%
    tidyr::pivot_wider(
      id_cols=c(method, computation, model),
      names_from = term,
      values_from=estimate
    ) %>%
    dplyr::rename(slope=`log(taxon_count)`, intercept=`(Intercept)`)
}
