import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# read  .csv fileinto a dataframe
house_data = pd.read_csv("data1/Housing.csv")
print(house_data)
area = house_data['area']
price = house_data['price']

print(area)

# machine learning handle arrays not data-frames
x = np.array(area).reshape(-1, 1)
y = np.array(price).reshape(-1, 1)

print(x)

# we use linear regression + fit() is the traning
model = LinearRegression()
model.fit(x,y)

# MSE and R value
regression_model_mse = mean_squared_error(x,y)
print("MSE: ", math.sqrt(regression_model_mse))
print("R squared value:", model.score(x,y))

# we can get the b values after the model fit 
# this is the b1 value
print(model.coef_[0])
# this is b0 in our model
print(model.intercept_[0])

# visualize the dataset with the fitted model
plt.scatter(x, y, color='green')
plt.plot(x, model.predict(x), color='black')
plt.title("Linear Regression")
plt.xlabel("area")
plt.ylabel("price")
plt.show()

# predicting the prices
print("Prediction by the model: ",model.predict([[5560]]))
