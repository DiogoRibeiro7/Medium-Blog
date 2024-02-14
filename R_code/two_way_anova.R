# Define the levels for each factor
study_techniques <- c("Flashcards", "Summarization", "MindMaps")
study_environments <- c("Quiet", "BackgroundMusic", "CafeNoise")

# Create a data frame with all combinations of study technique and environment
study_design <- expand.grid(Technique = study_techniques, Environment = study_environments)

# Repeat each combination to simulate multiple students per group
students_per_group <- 30
study_design <- study_design[rep(seq_len(nrow(study_design)), each = students_per_group), ]

# Generate random test scores for each student
set.seed(123) # For reproducibility
study_design$TestScore <- round(runif(length(study_design$Technique), min = 60, max = 100), digits = 0)

# Now, the corrected dataset 'study_design' includes the 'TestScore' variable correctly
# Let's try plotting again

library(ggplot2)

ggplot(study_design, aes(x = Technique, y = TestScore, fill = Environment)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Test Scores by Study Technique and Environment",
       x = "Study Technique",
       y = "Test Score") +
  theme(plot.title = element_text(hjust = 0.5))

write.csv(study_design,file='study_data.csv')

study_data <- read.csv("study_data.csv")

str(study_data)
summary(study_data)


ggplot(study_data, aes(x = Technique, y = TestScore, fill = Environment)) +
  geom_boxplot() +
  facet_wrap(~Environment) +
  theme_minimal() +
  labs(title = "Test Scores by Study Technique and Environment",
       x = "Study Technique",
       y = "Test Score")

shapiro.test(residuals(aov(TestScore ~ Technique * Environment, data = study_data)))

library(car)
leveneTest(TestScore ~ Technique * Environment, data = study_data)

