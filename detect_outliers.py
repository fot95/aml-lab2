# TODO: README: I have tried 2 ways:
# 1. I split the dataset into 5 folds, and at each iteration i keep 4 folds to train and 1 to validate.
#    At each iteration i do feature selection of 200 features, generate a new model, train and validate it.
#    After these iterations, I keep the features that the 5 SelectKBest have selected and I train again a model. Then I test it at the test set.

# 2. I split the dataset into 5 folds, and at each iteration i keep 4 folds to train and 1 to validate.
#    At each iteration i do feature selection of 200 features, generate a new model, train, validate it and test it at the test set.
#    After these iterations, I keep the mean value for the 5 generated prediction sets.
#    I don't know if it is correct :)))) 


import pandas as pd
import sys
import numpy
import copy

from sklearn.utils import column_or_1d
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV, Lasso, SGDRegressor, ElasticNet
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest
from sklearn.impute import SimpleImputer
#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import RFE, RFECV
from statistics import mean
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import ExtraTreeClassifier

from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

numpy.set_printoptions(threshold=sys.maxsize)


def sort(val):
    return val[1]  # sort using the 2nd element


def print_histogram(d):
	pos = numpy.arange(len(d.keys()))
	width = 1.0     # gives histogram aspect to the bar diagram

	ax = plt.axes()
	ax.set_xticks(pos + (width / 2))
	ax.set_xticklabels(d.keys())

	plt.bar(d.keys(), d.values(), width, color='g')

	plt.show()
	
def detect_outliers(train_data):
	outliers_list = []
	for i in range(1, 1000):
		op = IsolationForest( n_estimators=150
                    , max_samples=1000
                    , contamination='auto'
                    , n_jobs=-1
                    , behaviour='new')
		outliers_predict = op.fit_predict(train_data)
		print("------------------- Isolation Forest ", i)
		outliers = 0
		outliers_id = []
		for i in range(1, len(outliers_predict)):
			o = outliers_predict[i]
			if o == -1:
				outliers += 1
				outliers_id.append(i)
		outliers_list.append(outliers_id)

	my_dict = {}
	results=[]
	for i in range(0, x_train.shape[0]):
		my_dict[i] = 0
	for l in outliers_list:
		for i in l:
			my_dict[i]+=1

	my_dict_s = sorted(my_dict.items(), key=lambda kv: kv[1])
	print(my_dict_s)

	for i in my_dict.keys():
		if my_dict[i] > 900:
			results.append(i)

	#print_histogram(d)
	return results
	

x_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

y_train = y_train.drop('id', axis=1)
x_train = x_train.drop('id', axis=1)

scaler = StandardScaler()
x_train_new = scaler.fit_transform(x_train)
cols = list(x_train.columns.values)
x_train = pd.DataFrame(data=x_train_new, columns=cols)

#result = detect_outliers(x_train)
result = [(2915, 924), (3057, 926), (3713, 927), (4548, 937), (1980, 951), (2705, 958), (202, 960), (1127, 965), (3817, 966), (1288, 968), (3039, 971), (2063, 976), (1634, 980), (3177, 983), (3710, 984), (1144, 985), (3668, 985), (489, 986), (3487, 988), (4362, 989), (2024, 992), (1352, 993), (2093, 994), (1805, 995), (1456, 996), (4358, 996), (498, 997), (2039, 997), (758, 998), (1040, 998), (1397, 998), (3541, 998), (247, 999), (390, 999), (461, 999), (529, 999), (1184, 999), (1194, 999), (1789, 999), (1792, 999), (1886, 999), (2323, 999), (2842, 999), (2895, 999), (3131, 999), (4252, 999), (4298, 999), (4480, 999), (4646, 999)]

#result = (2892, 902), (4390, 903), (3057, 907), (2915, 913), (3713, 922), (4548, 933), (2705, 951), (1980, 960), (3039, 960), (1127, 962), (3817, 964), (202, 966), (1634, 972), (1144, 973), (2063, 977), (1288, 979), (3177, 979), (4362, 983), (3668, 984), (489, 985), (3487, 992), (3710, 993), (1352, 995), (498, 996), (758, 996), (1805, 997), (2024, 997), (2039, 997), (2093, 997), (4358, 997), (247, 998), (1184, 998), (1397, 998), (1456, 998), (3541, 998), (390, 999), (461, 999), (529, 999), (1040, 999), (1194, 999), (1789, 999), (1792, 999), (1886, 999), (2323, 999), (2842, 999), (2895, 999), (3131, 999), (4252, 999), (4298, 999), (4480, 999), (4646, 999)
outliers=[]
for pair in result:
	outliers.append(pair[0])
outliers.sort()
print(outliers)
print(len(outliers))
