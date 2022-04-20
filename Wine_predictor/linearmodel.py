# Contains code for ML using Linear Regression
# y = m x + b
import pandas as pd
# sklearn for working with models and regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix

class model_fit():
    def __init__(self, x, y):
        self.X = x
        self.Y = y
        self.train_x =None
        self.train_y =None
        self.test_x =None
        self.test_y =None
        self.m = None
        self.b = None

    def split_data(self):
        # Function to split data into test and train data
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        self.train_x = pd.DataFrame(self.train_x)
        self.test_x = pd.DataFrame(self.test_x)
        self.train_x.values.reshape(-1,1)
        self.test_x.values.reshape(-1,1)
        return self.train_x, self.test_x, self.train_y, self.test_y

    def run_regression(self):
        # Run model fitting
        print ("hi")   