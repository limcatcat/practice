import pandas as pd  # load and manipulate data, one-hot encoding
import numpy as np  # data manipulation
import matplotlib.pyplot as plt  # matplotlib is for drawing graphs
import matplotlib.colors as colors
from sklearn.utils import resample  # downsample the dataset
# split data into training and testing sets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale  # scale and center data
# this will make a support vector machine for classification
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV  # this will do cross validation
from sklearn.metrics import confusion_matrix  # this creates a confusion matrix
from sklearn.decomposition import PCA  # to perform PCA to plot the data


# import data
df = pd.read_csv("default of credit card clients.csv", header=1, sep=',')

print(df.head())

# rename a column
df.rename({'default payment next month': "DEFAULT"}, axis=1, inplace=True)
# axis=columns is necessary to make sure that you're changing the column name, not the row index
# inplace=True makes sure that the change happens inside the df, not creating a new copy of it

print(df.columns)
print(df.head())
