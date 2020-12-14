import tensorflow
import keras
import sklearn
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

#using dataset via the load_boston function
bostonData = load_boston()

'''
data = other factors (independent variables)
feature_names = column names
target = price (dependent variables)
'''

#convert the dataset to a dataframe
dataFrame_x = pd.DataFrame(bostonData.data, columns=bostonData.feature_names)
dataFrame_y = pd.DataFrame(bostonData.target)

#initialize linear regression model
linear = linear_model.LinearRegression()

#split the data into 67% training and 33% testing data
x_train, x_test, y_train, y_test = train_test_split(dataFrame_x, dataFrame_y, test_size=0.33, random_state=42)

#train the model with training data
linear.fit(x_train, y_train)

#accurary of the model
accuracy = linear.score(x_test, y_test)
print("Accuracy:", round(accuracy*100, 2), "%", "\n")

#print predictions based off test data
pricePredict = linear.predict(x_test)
print("Predicted values:")
for i in range(len(pricePredict)):
    print(pricePredict[i])

#checking model performance/accuracy using Mean Squared Error (MSE)
print("\nMSE:", np.mean((pricePredict - y_test) ** 2)[0])
