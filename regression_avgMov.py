import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from operator import itemgetter 

# to smooth the data
def movAvg(X, window):
    newX = X
    for i in range(window, len(X)-window):
        temp = X[i]
        for j in range(1, window+1):
            temp = temp + X[i-j] + X[i+j]
        temp = temp/(2*window + 1)
        newX[i] = temp
    return newX

# set the directory of the file to our current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_filename = '0601.csv'

with open(csv_filename, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    csv_contents = [_ for _ in csv_reader]
    if len(csv_contents) < 91:
        raise ValueError("CSV not long enough to process data.")

# get the numeric data from csv (header removed, index can be directed used)
# i = index of frame
def data(i, n):
    return float(csv_contents[i][n])

# 5 inputs, angle_x/y, head position x/y/z
# 11, 12 | 294 - 295
anglex = []
angley = []
headpox = []
headpoy = []
headpoz = []
headrox = []
headroy = []
headroz = []
# for test.csv, 342, 7326
for i in range(104, 4952): # these are dataset-specific numbers
    anglex.append(data(i, 11)) 
    angley.append(data(i, 12))
    headpox.append(data(i, 293))    # index in output: 293
    headpoy.append(data(i, 294))    # index in output: 294
    headpoz.append(data(i, 295))    # index in output: 295
    headrox.append(data(i, 296))
    headroy.append(data(i, 297))
    headroz.append(data(i, 298))
print(len(anglex), " datapoints have been recorded.")

# # get the range of data to make the histogram
# print("The min of angley is: ", min(angley))
# print("The max of angley is: ", max(angley))

# # make a histogram of anglex to see its distribution
# plt.hist(angley, bins=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
# plt.title("Histogram of angle_y")
# plt.show()

# read the outputs
def json_to_arr():
    pixelx = []
    pixely = []
    with open('./0601.json') as f:
        json_dict = json.load(f)
    #print(json_dict['perimeter_experiment']['points'][0])
    for point in json_dict['zooming_experiment']['points']:  # loop through list
        pixelx.append(point[0])
        pixely.append(point[1])

    return pixelx, pixely

pixelx, pixely = json_to_arr()

# smooth the data before training
window = 9
anglex = movAvg(anglex, window)
angley = movAvg(angley, window)
headpox = movAvg(headpox, window)
headpoy = movAvg(headpoy, window)
headpoz = movAvg(headpoz, window)
headrox = movAvg(headrox, window)
headroy = movAvg(headroy, window)
headroz = movAvg(headroz, window)

df = pd.DataFrame({'x_1': anglex, #x1
                    'x_2': angley, #x2
                    'x_3': headpox, #x3
                    'x_4': headpoy, #x4
                    'x_5': headpoz, #x5
                    'x_6': headrox,
                    'x_7': headroy,
                    'x_8': headroz,
                    'y_1': pixelx, #y1
                    'y_2': pixely}) #y2

x, y = df[["x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7", "x_8"]], df["y_2"]


# the range is the polynomial degree that will be tried during one run. Now it is only running on 5
for i in range(2, 6):
    poly = PolynomialFeatures(degree=i, include_bias=False)
    poly_features = poly.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.2, random_state=42) # get the training set and testing set

    poly_reg_model = LinearRegression()
    poly_reg_model.fit(x_train, y_train)
    # print('Coefficients of x are', poly_reg_model.coef_)
    # print('Intercept is', poly_reg_model.intercept_)

    poly_reg_y_predicted = poly_reg_model.predict(x_test)
    poly_reg_rmse = np.sqrt(mean_squared_error(y_test, poly_reg_y_predicted))
    r2 = r2_score(y_test, poly_reg_y_predicted)
    print('RMSE of degree', i, ' is ', poly_reg_rmse) # printing the errors
    print('R2 score of degree ', i, ' is ', r2)

# plt.plot(y_test, "b.")
# plt.plot(poly_reg_y_predicted, "r.", label = "prediction")
# plt.xlabel("$x_1$", fontsize = 18)
# plt.ylabel("$y$", rotation = 0, fontsize = 18)
# plt.legend(loc ="upper left", fontsize = 14)
# plt.title("Linear_Regression_predictions_plot")
# plt.show()

# lin_reg = LinearRegression()
# lin_reg.fit(x_poly, y)
# print('Coefficients of x are', lin_reg.coef_)
# print('Intercept is', lin_reg.intercept_)


# x_new = np.linspace(-3, 4, 100).reshape(100, 1)
# x_new_poly = poly_features.transform(x_new)
# y_new = lin_reg.predict(x_new_poly)
# plt.plot(x, y, "b.")
# plt.plot(x_new, y_new, "r-", linewidth = 2, label ="Predictions")
# plt.xlabel("$x_1$", fontsize = 18)
# plt.ylabel("$y$", rotation = 0, fontsize = 18)
# plt.legend(loc ="upper left", fontsize = 14)
  
# plt.title("Quadratic_predictions_plot")
# plt.show()