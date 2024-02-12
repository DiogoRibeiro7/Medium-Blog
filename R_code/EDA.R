# Load necessary libraries
library(ISLR) # For the dataset
library(ggplot2) # For data visualization
library(dplyr) # For data manipulation
library(nortest) # For normality tests


# Load the Carseats dataset
data("Carseats")

# Explore the dataset
dim(Carseats) # Dimensions of the dataset
str(Carseats) # Structure of the dataset
summary(Carseats) # Summary statistics


# Check for missing values
sum(is.na(Carseats))


suppressWarnings(RNGversion("3.5.0"))
set.seed(123)
Carseats[sample(seq(NROW(Carseats)), 20), "Income"] <- NA

suppressWarnings(RNGversion("3.5.0"))
set.seed(456)
Carseats[sample(seq(NROW(Carseats)), 10), "Urban"] <- NA


# Summary statistics for numerical variables
summary(Carseats[ ,sapply(Carseats, is.numeric)])

# Summary for categorical variables
summary(Carseats[ ,sapply(Carseats, is.factor)])

# Histogram of Sales
ggplot(Carseats, aes(x=Sales)) + geom_histogram(binwidth=1, fill="blue", color="black") + ggtitle("Distribution of Sales")


# Box plot for Sales across different locations
ggplot(Carseats, aes(x=Urban, y=Sales, fill=Urban)) + geom_boxplot() + ggtitle("Sales by Urban vs. Non-Urban Stores")

# Scatter plot of Sales vs. Advertising
ggplot(Carseats, aes(x=Advertising, y=Sales)) + geom_point() + geom_smooth(method="lm", color="red") + ggtitle("Sales vs. Advertising")

# Compute correlations
correlations <- cor(Carseats[ ,sapply(Carseats, is.numeric)])
print(correlations)

# Visualize correlations (optional, requires corrplot package)
library(corrplot)
corrplot(correlations, method="circle")


# Q-Q plot for the 'Sales' variable
qqnorm(Carseats$Sales, main = "Q-Q Plot for Sales")
qqline(Carseats$Sales, col = "red")

# Shapiro-Wilk test for the 'Sales' variable
shapiro.test(Carseats$Sales)

# Histogram with density plot for Sales
ggplot(Carseats, aes(x=Sales)) +
  geom_histogram(aes(y=..density..), binwidth=1, colour="black", fill="skyblue") +
  geom_density(alpha=.2, fill="#FF6666") +
  ggtitle("Histogram and Density Plot for Sales")

# Function to create histogram, density plot, and Q-Q plot for a given variable
plot_normality <- function(data, variable_name) {
  # Histogram and Density Plot
  ggplot(data, aes_string(x=variable_name)) +
    geom_histogram(aes(y=..density..), binwidth=1, colour="black", fill="skyblue") +
    geom_density(alpha=.2, fill="#FF6666") +
    ggtitle(paste("Histogram and Density Plot for", variable_name))
  
  # Q-Q Plot
  qqnorm(data[[variable_name]], main=paste("Q-Q Plot for", variable_name))
  qqline(data[[variable_name]], col="red")
}

# Example usage for 'Sales'
plot_normality(Carseats, "Sales")

# Relation between US and Sales

# Summary statistics of Sales by US location
aggregate(Sales ~ US, data = Carseats, FUN = function(x) c(mean = mean(x), sd = sd(x)))

# Box plot of Sales by US location
ggplot(Carseats, aes(x = US, y = Sales, fill = US)) +
  geom_boxplot() +
  ggtitle("Sales Distribution by US Location") +
  xlab("Store Location in the US") +
  ylab("Sales")

# Density plot of Sales by US location
ggplot(Carseats, aes(x = Sales, fill = US)) +
  geom_density(alpha = 0.5) + # Adjust transparency with alpha
  scale_fill_manual(values = c("Yes" = "blue", "No" = "red")) + # Custom colors for US and non-US
  ggtitle("Density Plot of Sales by US Location") +
  xlab("Sales") +
  ylab("Density") +
  theme_minimal() # Using a minimal theme for a cleaner look


# Welch's t-test for difference in Sales between US and non-US stores
t.test(Sales ~ US, data = Carseats, alternative = "two.sided", var.equal = FALSE)

# Create a contingency table of ShelveLoc and US
contingency_table <- table(Carseats$ShelveLoc, Carseats$US)
print(contingency_table)

# Perform the chi-squared test
chi_squared_test <- chisq.test(contingency_table)
print(chi_squared_test)

# Fit a simple linear regression model
model <- lm(Sales ~ Price, data=Carseats)

# Display the summary of the model
summary(model)


# Visualize the regression line
ggplot(Carseats, aes(x=Price, y=Sales)) +
  geom_point() + # Add data points
  geom_smooth(method="lm", color="blue") + # Add the regression line
  ggtitle("Relationship between Price and Sales") +
  xlab("Price") +
  ylab("Sales")

# Fit a one-way ANOVA model
anova_model <- aov(Sales ~ ShelveLoc, data = Carseats)

# Display the summary of the ANOVA model
summary(anova_model)

# Assuming the p-value was found to be significant upon manual check
TukeyHSD(anova_model)


ggplot(Carseats, aes(x = ShelveLoc, y = Sales, fill = ShelveLoc)) +
  geom_boxplot() +
  ggtitle("Sales by Shelf Location") +
  xlab("Shelf Location") +
  ylab("Sales")

