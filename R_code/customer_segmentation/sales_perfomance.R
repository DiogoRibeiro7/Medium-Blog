# Load necessary libraries
library(ggplot2)
library(dplyr)

# Read the data
data <- read.csv('data.csv', stringsAsFactors = FALSE)

# Convert InvoiceDate to Date type and create a TotalSales column
data$InvoiceDate <- as.POSIXct(data$InvoiceDate, format="%m/%d/%Y %H:%M")
data$Date <- as.Date(data$InvoiceDate)

# Create a TotalSales column
data$TotalSales <- data$Quantity * data$UnitPrice

# Aggregate data by Country
sales_by_country <- data %>%
  group_by(Country) %>%
  summarise(TotalSales = sum(TotalSales),
            AverageSalesPerOrder = mean(TotalSales),
            Orders = n_distinct(InvoiceNo)) %>%
  arrange(desc(TotalSales))

# Plot total sales by country
ggplot(sales_by_country, aes(x=reorder(Country, TotalSales), y=TotalSales)) +
  geom_bar(stat="identity", fill="steelblue") +
  coord_flip() +
  labs(title="Sales Performance by Country",
       x="Country",
       y="Total Sales") +
  theme_minimal()

# Additionally, plotting average sales per order might give more insights
ggplot(sales_by_country, aes(x=reorder(Country, AverageSalesPerOrder), y=AverageSalesPerOrder)) +
  geom_bar(stat="identity", fill="darkgreen") +
  coord_flip() +
  labs(title="Average Sales Per Order by Country",
       x="Country",
       y="Average Sales Per Order") +
  theme_minimal()
