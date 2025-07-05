# The imports
import numpy
import pandas as pd
import yfinance as yf
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
# Outputting the price table of a certain stock, which is currently Apple
stock = 'AAPL'

data_for_stock = yf.Ticker(stock)
data_for_stock = data_for_stock.history(period="10y")

del data_for_stock["Dividends"]
data_for_stock["Tomorrow"] = data_for_stock["Close"].shift(-1)

data_for_stock
# The below is to plot the stocks over time.
plt.figure(figsize=(14,5))
sns.set_style("ticks")
sns.lineplot(data=data_for_stock,x="Date",y='Close',color='green')
sns.despine()
plt.title("The Stock Price",size='x-large',color='black')

# Creating the training and testing data
import sklearn
testing_percentage = 0.025
train_d = data_for_stock[:int(-1*len(data_for_stock)*(testing_percentage))]
test_d = data_for_stock[int(-1*len(data_for_stock)*testing_percentage):]

plt.figure(figsize=(24,5))
sns.set_style("ticks")
sns.lineplot(data=train_d,x="Date",y='Close',color='firebrick')
sns.lineplot(data=test_d,x="Date", y='Close', color='blue')
sns.despine()
plt.title("The Training Data",size='x-large',color='black')

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=1)
predictors = ["Open", "High", "Low", "Volume"]
model = model.fit(train_d[predictors], train_d["Close"])
from sklearn.metrics import explained_variance_score

predictions = model.predict(test_d[predictors])

count = 0
print("Predicted\t\tActual")

for i in range(len(test_d)):
    print(str(predictions[i]) + "\t" + str(test_d["Close"][i]))
    if (abs(test_d["Close"][i] - predictions[i]) <= 0.01 * test_d["Close"][i]):
        count += 1
print("\n" + str(count / len(test_d)))
explained_variance_score(test_d["Close"], predictions)