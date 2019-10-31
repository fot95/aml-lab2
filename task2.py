import pandas as pd
import sys
import numpy
import copy

from sklearn.utils import column_or_1d
from sklearn.model_selection import GridSearchCV, cross_validate
#from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression
from sklearn.feature_selection import RFE, RFECV
from statistics import mean


numpy.set_printoptions(threshold=sys.maxsize)

def sort(val):
    return val[2]  # sort using the 2nd element


x_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

y_train = y_train.drop('id', axis=1)
x_train = x_train.drop('id', axis=1)

print(x_train.shape)
print(y_train.shape)

#1. Find how many elements from the train set belong to each class

classes = [0, 1, 2]
classes_pop = [0, 0, 0]
for i in list(y_train['y']):
	classes_pop[i] += 1
print('Samples/class: ', classes_pop)

#2. Check for missing values
print ('Missing values: ' , sum(list(x_train.isnull().sum())))


