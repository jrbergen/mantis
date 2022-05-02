# import standard libraries

from __future__ import annotations
from turtle import color

# pandas import already in dataset
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

from sklearn.linear_model import LinearRegression

# import custom data types and libraries
import dataset
import linearmodel


def AssignXY(wdata):
    # Function to allocate for X and Y values
    # for linear regression
    wdata = wdata.rename(columns={"volatile acidity" : "volatile_acidity"})
    X = wdata.volatile_acidity
    Y = wdata.quality
    print(X)
    plt.xlabel("Volatile Acidity")
    plt.ylabel("Quality")
    plt.title("Raw Data Plot of Quality vs Volatile Acidity")
    plt.grid()
    sn.scatterplot(X, Y)  # Plot and
    plt.show()  # Visualize Data
    return X, Y


def main():
    print("running")
    winedata = dataset.WineData()  # Create object for operating on data
    # print(winedata.path)

    winedata.import_data()  # Imported Data from csv format
    # print(winedata.df)                 # winedata.df stores all data

    X, Y = AssignXY(winedata.df)  # Obtaining X and Y for curve fitting
    curvefit = linearmodel.ModelClass(X, Y)  # Create object for curve fitting using intitializing with X and Y

    train_x, test_x, train_y, test_y = curvefit.split_data()
    model = LinearRegression()
    model.fit(train_x, train_y)
    print("Fitting Linear Model \n")
    #test_x, test_y = curvefit.get_test_data()
    print(test_x)
    pred = model.predict(test_x)
    test_x, test_y = curvefit.get_test_data()
    # print(np.sqrt(metrics.mean_squared_error(test_y, pred)))
    plt.xlabel("Volatile Acidity")
    plt.ylabel("Quality")
    plt.title("Test Data Plot of Quality vs Volatile Acidity")
    plt.grid()
    sn.scatterplot(test_x, pred)
    plt.show()

    print("Intercept: ", model.intercept_)
    print("Coefficient: ", model.coef_)
    b = model.intercept_
    m = model.coef_

    sn.distplot(test_y - pred)
    plt.show()
    # round grades (predictions)
    for i in range(len(pred)):
        pred[i] = round(pred[i])

    plt.xlabel("Volatile Acidity")
    plt.ylabel("Quality")
    plt.title("Quality vs Volatile Acidity")
    # obtain m (slope) and b(intercept) of linear regression line
    # m, b = np.polyfit(test_x, test_y, 1)
    sn.scatterplot(X, Y, label="Raw Data")
    # add linear regression line to scatterplot
    plt.grid()
    plt.plot(test_x, m * test_x + b, label="Linear Regression", color='r')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("Running")
    main()
    # run something
