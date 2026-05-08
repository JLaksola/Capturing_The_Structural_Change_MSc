library(dplyr)
library(lubridate)
library(brms)

#### Build priors
# Pooled priors
build_priors_from_window <- function(df, train_end, window_months = 360) {
  prior_start <- train_end %m-% months(window_months)
  
  prior_window <- df %>%
    filter(Date > prior_start, Date <= train_end)
  
  if (nrow(prior_window) < 50) {
    stop("Not enough data in prior_window to estimate priors.")
  }
  
  lm_fit <- lm(Real_Return_10Y ~ CAPE, data = prior_window)
  summ   <- summary(lm_fit)
  beta_tab <- summ$coefficients
  
  intercept_mean <- beta_tab["(Intercept)", "Estimate"]
  intercept_sd   <- beta_tab["(Intercept)", "Std. Error"]
  
  slope_mean <- beta_tab["CAPE", "Estimate"]
  slope_sd   <- beta_tab["CAPE", "Std. Error"]
  
  # Build the prior strings
  slope_prior_str     <- paste0("normal(", slope_mean, ", ", slope_sd, ")")
  intercept_prior_str <- paste0("normal(", intercept_mean, ", ", intercept_sd, ")")
  priors <- c(
    do.call(prior,list(slope_prior_str, class = "b", coef = "CAPE")),
    do.call(prior,list(intercept_prior_str, class = "b", coef = "Intercept"))
    )
  
  priors
}


# Hierarchical priors
hierarchical_priors_from_window_uncentered <- function(df, train_end, window_months = 360) {
  prior_start <- train_end %m-% months(window_months)
  
  prior_window <- df %>%
    dplyr::filter(Date > prior_start, Date <= train_end)
  
  if (nrow(prior_window) < 50) {
    stop("Not enough data in prior_window to estimate priors.")
  }
  
  lm_fit   <- lm(Real_Return_10Y ~ CAPE, data = prior_window)
  summ     <- summary(lm_fit)
  beta_tab <- summ$coefficients
  
  intercept_mean <- beta_tab["(Intercept)", "Estimate"]
  intercept_sd   <- beta_tab["(Intercept)", "Std. Error"]
  slope_mean     <- beta_tab["CAPE",        "Estimate"]
  slope_sd       <- beta_tab["CAPE",        "Std. Error"]
  
  slope_sd      <- slope_sd * 2
  intercept_sd  <- intercept_sd * 2
  
  slope_prior_str     <- paste0("normal(", slope_mean,     ", ", slope_sd,     ")")
  intercept_prior_str <- paste0("normal(", intercept_mean, ", ", intercept_sd, ")")
  
  priors <- c(
    do.call(prior,list(slope_prior_str, class = "b", coef = "CAPE")),
    do.call(prior,list(intercept_prior_str, class = "b", coef = "Intercept")),
    prior(exponential(0.25), class = "sd", group = "Period30", coef = "Intercept"),
    prior(exponential(1),   class = "sd", group = "Period30", coef = "CAPE"),
    prior(lkj(2), class = "cor", group = "Period30")
  )
  
  priors
}



### Utility functions ###

# Function for generating posterior predictions
generate_prediction <- function(model, newdata, prob = 0.95){
  posterior_draws <- posterior_epred(
    model,
    newdata = newdata,
    allow_new_levels = TRUE
  )
  pred_mean <- mean(posterior_draws)
  ci_lower <- quantile(posterior_draws, probs = (1 - prob) / 2)
  ci_upper <- quantile(posterior_draws, probs = 1 - (1 - prob) / 2)
  return (c(pred_mean, ci_lower, ci_upper))
}

# Store convergence diagnostics
convergence_diagnostics <- function(current_date, model){
  # Posterior parameter draws
  draws_array <- as_draws_array(model)
  
  # Automatically select parameter columns (exclude metadata/internal columns)
  param_cols <- setdiff(dimnames(draws_array)[[3]], c("lp__", "lprior"))
  
  # Initialize vectors to store diagnostics
  rhat_vals <- numeric(length(param_cols))
  ess_bulk_vals <- numeric(length(param_cols))
  ess_tail_vals <- numeric(length(param_cols))
  
  # Loop over parameters
  for (i in seq_along(param_cols)) {
    par <- param_cols[i]
    par_draws <- draws_array[,,par]  # iterations x chains
    rhat_vals[i] <- rhat(par_draws)
    ess_bulk_vals[i] <- ess_bulk(par_draws)
    ess_tail_vals[i] <- ess_tail(par_draws)
  }
  
  # Combine into data frame
  diag_df <- data.frame(
    Date      = current_date,
    Parameter = param_cols,
    Rhat      = rhat_vals,
    ESS_Bulk  = ess_bulk_vals,
    ESS_Tail  = ess_tail_vals
  )
  
  return (diag_df)
}

# Compute lpd
compute_log_pred_density_hier <- function(model, newdata) {
  log_lik_matrix <- log_lik(model, newdata = newdata, allow_new_levels = TRUE)
  lik_matrix     <- exp(log_lik_matrix)
  log(colMeans(lik_matrix))
}










