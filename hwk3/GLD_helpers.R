# assumes you have already done rstan::expose_stan_functions("quantile_functions.stan")
GLD_solver <- function(lower_quartile, median, upper_quartile, other_quantile, alpha) {
  a_s <- find_chi_xi(c(lower_quartile, median, upper_quartile, other_quantile, alpha), 
                     unbounded = FALSE)
  names(a_s) <- c("asymmetry", "steepness")
  low <- GLD_icdf(0, median, IQR = upper_quartile - lower_quartile,
                  asymmetry = a_s[1], steepness = a_s[2])
  if (low > -Inf) warning("solution implies a bounded lower tail at ", low)
  high <- GLD_icdf(1, median, IQR = upper_quartile - lower_quartile, 
                   asymmetry = a_s[1], steepness = a_s[2])
  if (high < Inf) warning("solution implies a bounded upper tail at ", high)
  return(a_s)
}
