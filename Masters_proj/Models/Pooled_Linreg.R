rm(list = ls())
library(lubridate)
library(ggplot2)
library(dplyr)
library(brms)
library(purrr)
library(parallel)
library(posterior)
library(tidyr)
library(cmdstanr)
setwd("C:/Users/Käyttäjä/Desktop/BDA_project/models")
source("C:/Users/Käyttäjä/Desktop/BDA_project/models/Functions.R")
set_cmdstan_path("~/cmdstan/cmdstan-2.37.0")

cores <- max(1, parallel::detectCores() - 1)

### Preprocess data ###
file_path <- "C:/Users/Käyttäjä/Desktop/BDA_project/data/Shiller_cleaned.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE) %>%
  mutate(Date = as.Date(Date)) %>%
  filter(complete.cases(.)) %>%
  arrange(Date)

train_start <- as.Date("1881-01-01")
test_start <- as.Date("1990-01-01")
test_end <- as.Date("2015-09-01")

# Lists to store the results and diagnostics
predictions <- c()
actuals <- c()
lowers <- c()
uppers <- c()
lpds <- c()
dates_vec <- as.Date(character())
diagnostics <- list()

# Generate test dates
test_dates <- seq(from = test_start, to = test_end, by = "month")
n_iter <- length(test_dates)
step_size_months <- 6L


### Define the model ###
formula <- bf(
  Real_Return_10Y ~ 1 + CAPE,
  family = "gaussian",
  center = FALSE
)


### Rolling Forecast Loop With Model Updates ###
# Define the updating schedule for the priors
prior_step_years <- 3L

prior_update_dates <- seq(
  from = test_start,
  to = test_end,
  by = paste0(prior_step_years, " years")
)

# List to store priors per block
priors_used <- vector("list", length(prior_update_dates))


### The main loop over different prior blocks ###
block_index <- 0

for (k in seq_along(prior_update_dates)) {
  
  prior_date <- as.Date(prior_update_dates[k])
  
  # Training end for the first test date in this block
  train_end <- as.Date(prior_date) %m-% years(10) %m-% months(1)
  train <- df %>%
    filter(Date >= train_start, Date <= train_end)
  
  current_priors <- build_priors_from_window(
    df = df,
    train_end = train_end,
    window_months = 120
  )
  
  # Save the priors to a list
  priors_used[[k]] <- list(
    block = k,
    prior_date = prior_date,
    train_end = train_end,
    priors = current_priors
  )
  
  # The model for the current block
  model <- brm(
    formula = formula,
    prior = current_priors,
    data = train,
    chains = 3,
    iter = 2000,
    warmup = 1000,
    backend = "cmdstanr",
    cores = cores,
    adapt_delta = 0.99,
    max_treedepth = 15,
    seed = 1
  )
  
  
  # Block boundaries
  block_start <- as.Date(prior_date)
  block_end <- if (k < length(prior_update_dates)) {
    as.Date(prior_update_dates[k + 1]) %m-% months(1)
  } else {
    test_end
  }
  
  block_dates <- seq(
    from = block_start,
    to = block_end,
    by = paste0(step_size_months, " months")
  )
  
  # Inner loop for the current block
  for (current_date in block_dates) {
    
    current_date <- as.Date(current_date)
    
    block_index <- block_index + 1
    cat("\nBLOCK", k, "STEP", block_index,"DATE", as.character(current_date), "\n")
    
    # Define a 6-month test set window
    window_end <- min(current_date %m+% months(step_size_months - 1L),block_end)
    
    test_sample <- df %>%
      filter(Date >= current_date, Date <= window_end)
    
    if (nrow(test_sample) == 0) next  # just in case
    
    # Predictions
    posterior_draws <- posterior_epred(
      model,
      newdata = test_sample
    )
    
    # Store the results
    y_pred_vec <- colMeans(posterior_draws)
    ci_lower_vec <- apply(posterior_draws, 2, quantile, probs = 0.025)
    ci_upper_vec <- apply(posterior_draws, 2, quantile, probs = 0.975)
    y_true_vec <- test_sample$Real_Return_10Y
    
    # Log predictive density for each testing date
    lpd_vec <- compute_log_pred_density_hier(model, test_sample)
    
    # Store all results to the lists
    predictions <- c(predictions, as.numeric(y_pred_vec))
    actuals <- c(actuals, as.numeric(y_true_vec))
    lowers <- c(lowers, ci_lower_vec)
    uppers <- c(uppers, ci_upper_vec)
    dates_vec <- c(dates_vec, test_sample$Date)
    lpds <- c(lpds, lpd_vec)
    
    # Store the convergence diagnostics for the testing dates
    diag_df <- convergence_diagnostics(current_date, model)
    diag_df$Block <- k
    diag_df$Step  <- block_index
    diagnostics[[block_index]] <- diag_df
    
    # Update the training end date
    next_train_end <- window_end %m-% years(10) %m-% months(1)
    
    if (next_train_end > train_end) {
      train_end <- next_train_end
      train <- df %>%
        filter(Date >= train_start, Date <= train_end)
      
      model <- update(
        model,
        newdata = train,
        recompile = FALSE)
    }
  }
  # Save the current block's model
  saveRDS(
    model,
    file = paste0(
      "Pooled_50yr_block_",
      sprintf("%02d", k), "_",
      format(block_end, "%Y-%m-%d"),
      ".rds"
    )
  )
}

# Save the results
results_df <- data.frame(
  Date = dates_vec,
  Predicted = predictions,
  Actual = actuals,
  Upper = uppers,
  Lower = lowers,
  Lpds = lpds
)

###########
# Import data
# Set the wanted results folder
setwd("~/Desktop/BDA_project/Results/Pooled_50yr_prior")

# Forecast results
results_df <- read.csv("results_forecast.csv", stringsAsFactors = FALSE)
results_df$Date <- as.Date(results_df$Date)   # convert back to Date

# Converge diagnostics
diagnostics_df <- read.csv("diagnostics_forecast.csv", stringsAsFactors = FALSE)
diagnostics_df$Date <- as.Date(diagnostics_df$Date)

# Priors used in each block
priors_df <- read.csv("priors_used.csv", stringsAsFactors = FALSE)
priors_df$prior_date <- as.Date(priors_df$prior_date)
priors_df$train_end  <- as.Date(priors_df$train_end)

# The last fitted model
library(rstan)
model <- readRDS("Pooled_50yr_block_09_2015-09-01.rds")
###########


### Inspect results ###
overall_rmse <- sqrt(mean((results_df$Actual - results_df$Predicted)^2))
r_squared <- cor(results_df$Actual, results_df$Predicted)^2
total_elpd <- sum(results_df$lpds)
y  <- results_df$Actual
yh <- results_df$Predicted

rss <- sum((y - yh)^2)
tss <- sum((y - mean(y))^2)
r2_pred <- 1 - rss / tss

cat("Overall RMSE:", overall_rmse, "\n")
cat("Overall R-squared:", r_squared, "\n")
cat("Overall ELPD:", total_elpd, "\n")
cat("Overall OOS R^2:", r2_pred)

ggplot(results_df, aes(x = Date)) +
  geom_line(aes(y = Predicted, colour = "Predicted")) +
  geom_line(aes(y = Actual,    colour = "Actual")) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper),alpha = 0.1) +
  labs(
    title = "Predicted vs Actual Returns",
    x = "Date",
    y = "Returns",
    colour = ""
  ) +
  theme_minimal()


### Inspect diagnostics ###
summary(model)

param_diag_df <- bind_rows(diagnostics)

ggplot(param_diag_df, aes(x = Date, y = Rhat)) +
  geom_line() +
  facet_wrap(~ Parameter, scales = "free_y") +
  labs(
    title = "Rhat over time per parameter",
    x = "Date",
    y = "Rhat"
  ) +
  theme_minimal()

ess_long <- param_diag_df %>%
  pivot_longer(
    cols = c(ESS_Bulk, ESS_Tail),
    names_to = "ESS_Type",
    values_to = "ESS_Value"
  )

ggplot(ess_long, aes(x = Date, y = ESS_Value, linetype = ESS_Type)) +
  geom_line() +
  facet_wrap(~ Parameter, scales = "free_y") +
  labs(
    title = "ESS (Bulk and Tail) over time per parameter",
    x = "Date",
    y = "ESS",
    linetype = "Type"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

mcmc_plot(model, type="trace")

# Posterior predictive checks
pp_check(model)
pp_check(model, type = "scatter_avg")
pp_check(model, type = "intervals")
pp_check(model, type = "error_hist")
pp_check(model, type = "error_scatter")



### Save the diagnostics and results ###
diagnostics_df <- bind_rows(diagnostics)

# Results and diagnostics as csv
write.csv(results_df, "results_forecast.csv", row.names = FALSE)
write.csv(diagnostics_df, "diagnostics_forecast.csv", row.names = FALSE)

# Save the model summary
smry <- summary(model)

# Save the printed version
capture.output(
  print(smry),
  file = "model_summary.txt"
)

# Save the model priors
# Priors_used into a single data frame
priors_df <- map_dfr(priors_used, function(x) {
  if (is.null(x)) return(NULL)
  
  p_df <- as.data.frame(x$priors)
  
  p_df$block <- x$block
  p_df$prior_date <- x$prior_date
  p_df$train_end  <- x$train_end
  
  p_df
})

# Save the priors to a csv
write.csv(priors_df, "priors_used.csv", row.names = FALSE)
