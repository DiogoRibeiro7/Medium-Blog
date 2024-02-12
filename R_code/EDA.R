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
