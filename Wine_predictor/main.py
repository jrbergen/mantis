# import standard libraries

from __future__ import annotations

# pandas import already in dataset
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import metrics

#import custom data types and libraries
import dataset
import linearmodel

def AssignXY(wdata):
    # Function to allocate for X and Y values
    # for linear regression
    wdata = wdata.rename(columns={'fixed acidity': 'fixed_acidity'})
    X = wdata.fixed_acidity
    Y = wdata.quality
    print (X)
    sn.scatterplot(X,Y)     # Plot and 
    plt.show()              # Visualize Data
    return X, Y

def main():
    print("running")
    winedata = dataset.WineData()       # Create object for operating on data
    #print(winedata.path)
    
    winedata.import_data()               # Imported Data from csv format 
    #print(winedata.df)                 # winedata.df stores all data

    X, Y = AssignXY(winedata.df)        # Obtaining X and Y for curve fitting
    curvefit = linearmodel.model_fit(X, Y) # Create object for curve fitting using intitializing with X and Y

    train_x,test_x, train_y, test_y = curvefit.split_data()
    model = LinearRegression()
    model.fit(train_x, train_y)
    print("Fitting Linear Model \n")
    pred = model.predict(test_x)    # Predict y values based on test_x
    #print(np.sqrt(metrics.mean_squared_error(test_y, pred))) 
    sn.scatterplot(pred, test_y)
    plt.show()
    
    print("Intercept: ", model.intercept_)
    print("Coefficient: ", model.coef_)
    
    sn.distplot(test_y - pred)
    plt.show()
    # round grades (predictions)
    for i in range(len(pred)):
        pred[i] = round(pred[i])

    plt.xlabel('Prediction for Quality') 
    plt.ylabel('Y test') 
  
    # Formatting Graph
    plt.title("Sulfur linear regression")
    plt.plot(pred, test_y, 'o')

    #obtain m (slope) and b(intercept) of linear regression line
    m, b = np.polyfit(pred, test_y, 1)

    #add linear regression line to scatterplot 
    plt.plot(test_y, m*test_y+b)
    plt.show()


if __name__ == "__main__":
    print ("Running")
    main()
    #run something