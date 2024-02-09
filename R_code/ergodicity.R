library(ggplot2)
library(dplyr)

simulate_random_walk <- function(steps = 1000, trials = 1000) {
  # Generate random steps
  walks <- matrix(sample(c(-1, 1), size = steps * trials, replace = TRUE), nrow = trials)
  # Cumulative sum to find positions
  positions <- t(apply(walks, 1, cumsum))
  
  # Time average for the first trial
  time_avg_position <- mean(positions[1, ])
  
  # Ensemble average after all steps
  ensemble_avg_positions <- mean(positions[, steps])
  
  return(list(positions = positions, time_avg = time_avg_position, ensemble_avg = ensemble_avg_positions))
}

# Parameters
steps <- 1000
trials <- 1000

# Simulation
results <- simulate_random_walk(steps, trials)
positions <- results$positions
time_avg <- results$time_avg
ensemble_avg <- results$ensemble_avg

# Data preparation for plotting
plot_data <- data.frame(Step = 1:steps, Position = positions[1, ])

# Visualization using ggplot2
ggplot(plot_data, aes(x = Step, y = Position)) +
  geom_line(color = "blue", size = 1) +
  geom_hline(yintercept = time_avg, color = "red", linetype = "dashed", size = 1.2, 
             label = paste("Time Average:", round(time_avg, 2))) +
  geom_hline(yintercept = ensemble_avg, color = "green", linetype = "dotted", size = 1.2, 
             label = paste("Ensemble Average:", round(ensemble_avg, 2))) +
  labs(title = "Demonstration of Ergodicity in a One-Dimensional Random Walk",
       x = "Step", y = "Position") +
  theme_minimal()
