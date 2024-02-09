# Load necessary library
library(ggplot2)

# Set seed for reproducibility
set.seed(123)

# Generate sample data
x <- 1:100
# Introduce heteroscedasticity with increasing standard deviation
y <- x * 1.5 + rnorm(100, mean = 0, sd = sqrt(x))

# Fit a linear model
model <- lm(y ~ x)

# Create a dataframe of fitted values and residuals
model_data <- data.frame(fitted = fitted(model), residuals = resid(model))

# Plot residuals vs. fitted values to visualize heteroscedasticity
ggplot(model_data, aes(x = fitted, y = residuals)) +
  geom_point() +  # Add points
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +  # Add a horizontal line at 0
  labs(x = "Fitted Values", y = "Residuals", title = "Residuals vs. Fitted Values Plot") +
  theme_minimal()  # Use a minimal theme for a cleaner look
