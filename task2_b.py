import pandas as pd
import sys
import numpy
import copy

from sklearn.utils import column_or_1d
from sklearn.decomposition import PCA, NMF
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import chi2, f_regression, f_classif
from sklearn.feature_selection import RFE, RFECV
from statistics import mean
from sklearn.metrics import make_scorer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score

from imblearn.over_sampling import SMOTE
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline, Pipeline

from detect_outliers import detect_outliers

numpy.set_printoptions(threshold=sys.maxsize)

def sort(val):
    return val[2]  # sort using the 2nd element

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

def print_samples_per_class(y):
	classes = [0, 1, 2]
	classes_pop = [0, 0, 0]
	for i in list(y):
		classes_pop[i] += 1

	print('Samples/class: ', classes_pop)


def generate_model_report(y, y_pred):
	print("Confusion Matrix:")
	print(confusion_matrix(y, y_pred, labels=[0, 1, 2]))

	print("Classification Report:")
	print(classification_report(y, y_pred, labels=[0, 1, 2]))

	print("BMCA:")
	print(balanced_accuracy_score(y, y_pred)) 

def remove_outliers(x, y, outliers):
	outliers_predict = [1] * x.shape[0]
	for i in range(0, len(outliers)):
		outliers_predict[outliers[i]] = -1
	
	x['is_outlier'] = outliers_predict
	x = x[x.is_outlier != -1]
	x = x.drop('is_outlier', axis=1)

	y['is_outlier'] = outliers_predict
	y = y[y.is_outlier != -1]
	y = y.drop('is_outlier', axis=1)

	return x, y

def select_features(x, y):
	'''
	feature_selector = SelectKBest(f_classif, k=400)
	x_sel = feature_selector.fit_transform(x, y)
	mask = feature_selector.get_support()  # list of booleans
	new_features = []  # The list of your best features

	for bool, feature in zip(mask, cols):
		if bool:
			new_features.append(feature)
	x_new = pd.DataFrame(data=x_sel, columns=new_features)
	print("-------------------------------------------new_features size:", len(new_features))
	scores_l = list(feature_selector.scores_)
	scores_l.sort(reverse=True)
	print(scores_l)
	return x_new
	'''
	feature_selector = PCA(n_components=200)
	cols = list(x.columns.values)
	feature_selector.fit(x)
	x_new = feature_selector.transform(x)
	print(x_new.shape)

	x = pd.DataFrame(data=x_new, columns=cols[0:200])
	return x, feature_selector

# load dataset
x_train_init = pd.read_csv("X_train.csv")
y_train_init = pd.read_csv("y_train.csv")

y_train_init = y_train_init.drop('id', axis=1)
x_train_init = x_train_init.drop('id', axis=1)


outliers = [202, 247, 390, 461, 489, 498, 529, 758, 1040, 1127, 1144, 1184, 1194, 1288, 1352, 1397, 1456, 1634, 1789, 1792, 1805, 1886, 1980, 2024, 2039, 2063, 2093, 2323, 2705, 2842, 2895, 2915, 3039, 3057, 3131, 3177, 3487, 3541, 3668, 3710, 3713, 3817, 4252, 4298, 4358, 4362, 4480, 4548, 4646]

# data scaling
scaler = StandardScaler()
x_train_new = scaler.fit_transform(x_train_init)
cols = list(x_train_init.columns.values)
x_train_init = pd.DataFrame(data=x_train_new, columns=cols)

x_train_init, y_train_init = remove_outliers(x_train_init, y_train_init, outliers)

# feature selection
x_train_init, f_selector = select_features(x_train_init, y_train_init)


outliers_2 = [160, 331, 1279, 1815, 2093, 2834, 3154, 3580, 3841, 4372]
x_train_init, y_train_init = remove_outliers(x_train_init, y_train_init, outliers_2)


# split data into train and validation set
st = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
st.get_n_splits(x_train_init, y_train_init)

for train_index, test_index in st.split(x_train_init, y_train_init):
    x_train, x_val = x_train_init.iloc[train_index], x_train_init.iloc[test_index]
    y_train, y_val = y_train_init.iloc[train_index], y_train_init.iloc[test_index]

print(y_train.shape)
print_samples_per_class(y_train['y'])

print(y_val.shape)
print_samples_per_class(y_val['y'])

################################################################################################

#print('Using SMOTE:')
#smote = SMOTE('not majority')
#x_train, y_train = smote.fit_sample(x_train, y_train)
#print_samples_per_class(y_train)
clf = SVC(C=1.0, kernel='rbf', gamma=0.001, class_weight='balanced')
clf.fit(x_train, y_train)

'''
clf=SVC(kernel = "rbf", class_weight='balanced')

N_FEATURES_OPTIONS = [150, 200, 250, 300, 350]
C_OPTIONS = [1.0, 10.0, 100.0, 1000.0]
G_OPTIONS = [0.00001, 0.0001, 0.001, 0.01]
param_grid = [
    {
  
        #'reduce_dim': [SelectKBest(f_classif)],
        #'reduce_dim__k': N_FEATURES_OPTIONS,
        'class__C': C_OPTIONS,
		'class__gamma': G_OPTIONS
    }
]

pipe = Pipeline([('class', clf)])
gs = GridSearchCV(pipe, param_grid=param_grid, scoring=make_scorer(balanced_accuracy_score), cv=5, n_jobs=-1, refit=True, return_train_score=True, verbose=2)
gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
clf = gs.best_estimator_
'''

print('test at the validation set')
y_val_pred = clf.predict(x_val)
generate_model_report(y_val, y_val_pred)

print('test at the whole data set')
y_init_pred = clf.predict(x_train_init)
generate_model_report(y_train_init, y_init_pred)


################################################################################################
# Test Set
test_set = pd.read_csv("X_test.csv")
x_test = test_set.drop('id', axis=1)
# scaling
x_test_scaled = scaler.fit_transform(x_test)
cols = list(x_test.columns.values)
x_test = pd.DataFrame(data=x_test_scaled, columns=cols)
x_test = f_selector.transform(x_test)
y_test = clf.predict(x_test)
Id = test_set['id']
df = pd.DataFrame(Id)
df.insert(1, "y", y_test)
df.to_csv('solution.csv', index=False)
