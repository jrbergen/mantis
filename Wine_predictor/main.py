# import standard libraries

from __future__ import annotations

# pandas import already in dataset
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

# sklearn for working with models and regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

#import custom data types and libraries
import dataset

def AssignXY(wdata):
    # Function to allocate for X and Y values
    # for linear regression
    wdata = wdata.rename(columns={'fixed acidity': 'fixed_acidity'})
    X = wdata.density
    Y = wdata.quality
    print (X)
    sn.scatterplot(X,Y)     # Plot and 
    plt.show()              # Visualize Data


def main():
    print("running")
    winedata = dataset.WineData()       # Create object for operating on data
    #print(winedata.path)
    
    winedata.ImportData()               # Imported Data from csv format 
    #print(winedata.df)                 # winedata.df stores all data

    AssignXY(winedata.df)

if __name__ == "__main__":
    print ("Running")
    main()
    #run something