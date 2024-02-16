if (!requireNamespace("Ecdat", quietly = TRUE)) install.packages("Ecdat")
if (!requireNamespace("MASS", quietly = TRUE)) install.packages("MASS")
if (!requireNamespace("tidyverse", quietly = TRUE)) install.packages("tidyverse") # For zero-inflated and count models
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("AER", quietly = TRUE)) install.packages("AER")
if (!requireNamespace("broom", quietly = TRUE)) install.packages("broom")

library(Ecdat)
library(tidyverse)
library(MASS) # or library(lme4)
library(AER)
library(broom)

strikes_data <- read.csv("strikes.csv")

glimpse(strikes_data)
summary(strikes_data)

strikes_data <- na.omit(strikes_data) # Example for handling missing values

ggplot(strikes_data, aes(x = strikes)) + 
  geom_histogram(binwidth = 1, fill = "blue", color = "black") + 
  theme_minimal() + 
  labs(title = "Distribution of Strike Counts", x = "Strike Count", y = "Frequency")

ggplot(strikes_data, aes(x = output, y = strikes)) + 
  geom_point() + 
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs")) + 
  theme_minimal() + 
  labs(title = "Scatter Plot of Strikes vs. Output", x = "Output", y = "Strikes")

ggplot(strikes_data, aes(x = time, y = strikes)) + 
  geom_line() + 
  geom_smooth(se = FALSE, color = "red") + 
  theme_minimal() + 
  labs(title = "Strikes Over Time", x = "Time", y = "Strikes")

# Calculate the mean and variance of the count variable
mean_strikes <- mean(strikes_data$strikes)
var_strikes <- var(strikes_data$strikes)

# Print the results
cat("Mean of Strikes:", mean_strikes, "\nVariance of Strikes:", var_strikes, "\n")

# Calculate correlation matrix for numeric predictors
cor_matrix <- cor(strikes_data[, c("output", "time")])

# Print the correlation matrix
print(cor_matrix)

# Expected number of zeros under Poisson distribution
expected_zeros <- dpois(0, lambda = mean_strikes) * nrow(strikes_data)

# Observed number of zeros
observed_zeros <- sum(strikes_data$strikes == 0)

# Print the results
cat("Expected Zeros:", expected_zeros, "\nObserved Zeros:", observed_zeros, "\n")

