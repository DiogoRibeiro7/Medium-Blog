# Distribution of transactions by country
library(ggplot2)

# Read the data
data <- read.csv('data.csv', stringsAsFactors = FALSE)

data %>%
  group_by(Country) %>%
  summarise(Transactions = n()) %>%
  ggplot(aes(x = reorder(Country, Transactions), y = Transactions)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Transactions by Country", x = "Country", y = "Number of Transactions")

# Identifying cancelled orders
cancelled_orders <- data %>%
  filter(grepl("^C", InvoiceNo)) # Orders with an invoice number starting with 'C'

# Count of cancelled orders
cat("Number of cancelled orders:", nrow(cancelled_orders), "\n")

# Unique stock codes and their frequencies
stock_code_freq <- data %>%
  group_by(StockCode) %>%
  summarise(Frequency = n()) %>%
  arrange(desc(Frequency))

# Display the top 10 stock codes
head(stock_code_freq, 10)

# Adding a column for total price
data$TotalPrice <- data$Quantity * data$UnitPrice

# Calculating basket price by invoice
basket_price <- data %>%
  group_by(InvoiceNo) %>%
  summarise(BasketPrice = sum(TotalPrice))

# Displaying the summary of basket prices
summary(basket_price$BasketPrice)
