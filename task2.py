import pandas as pd
import sys
import numpy
import copy

from sklearn.utils import column_or_1d
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold
from sklearn.ensemble import IsolationForest
#from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression
from sklearn.feature_selection import RFE, RFECV
from statistics import mean
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score

from imblearn.over_sampling import SMOTE


numpy.set_printoptions(threshold=sys.maxsize)

def sort(val):
    return val[2]  # sort using the 2nd element

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

x_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

y_train = y_train.drop('id', axis=1)
x_train = x_train.drop('id', axis=1)

print(x_train.shape)
print(y_train.shape)

# 1. Find how many elements from the train set belong to each class

classes = [0, 1, 2]
classes_pop = [0, 0, 0]
for i in list(y_train['y']):
	classes_pop[i] += 1

print('Samples/class: ', classes_pop)

# 2. Check for missing values
print ('Missing values: ' , sum(list(x_train.isnull().sum())))

# 3. Some preprocessing - scaling - outliers detection - class balancing - feature selection
# Scaling

# Outliers detection
'''
op = IsolationForest( n_estimators=150
                    , max_samples=1000
                    , contamination='auto'
                    , max_features=1.0, bootstrap=False
                    , n_jobs=10
                    , behaviour='new'
                    , random_state=42
                    , verbose=0
                    , warm_start=False)
outliers_predict = op.fit_predict(x_train)
outliers = 0
for o in outliers_predict:
    if o == -1:
        outliers += 1

print('number of outliers:', outliers)

x_train['is_outlier'] = outliers_predict
x_train = x_train[x_train.is_outlier != -1]
x_train = x_train.drop('is_outlier', axis=1)

y_train['is_outlier'] = outliers_predict
y_train = y_train[y_train.is_outlier != -1]
y_train = y_train.drop('is_outlier', axis=1)
'''

# feature selection

feature_selector = PCA(n_components=100)
cols = list(x_train.columns.values)
feature_selector.fit(x_train)
x_train_sel = feature_selector.transform(x_train)
print(x_train_sel.shape)

x_train = pd.DataFrame(data=x_train_sel, columns=cols[0:100])

print('Using SMOTE:')
smote = SMOTE('not majority')
x_train_init = copy.deepcopy(x_train)
y_train_init = copy.deepcopy(y_train)

x_train, y_train = smote.fit_sample(x_train, y_train.y)
print(x_train.shape)
print(y_train.shape)

classes_pop = [0, 0, 0]
for i in list(y_train):
	classes_pop[i] += 1
print('Samples/class: ', classes_pop)

# 4. Classifier training + tuning
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_train)


print("Confusion Matrix of Training:")
print(confusion_matrix(y_train, y_pred, labels=[0, 1, 2]))

print("Classification Report of Training:")
print(classification_report(y_train, y_pred, labels=[0, 1, 2]))

print("BMCA:")
print(balanced_accuracy_score(y_train, y_pred))

val_scores = []
features = []
test_df=[]

N = 10
kf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)
for train_index, test_index in kf.split(x_train_init, y_train_init):
    X_train_i, X_val_i = x_train_init.iloc[train_index], x_train_init.iloc[test_index]
    Y_train_i, Y_val_i = y_train_init.iloc[train_index], y_train_init.iloc[test_index]

    smote = SMOTE('not majority')
    X_train_smote, Y_train_smote = smote.fit_sample(X_train_i, Y_train_i.y)

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train_smote, Y_train_smote)
    Y_pred_val = clf.predict(X_val_i)

    test_s = balanced_accuracy_score(Y_val_i, Y_pred_val)

    val_scores.append(test_s)

print(val_scores)
print("mean: ", mean(val_scores))
print("std: ", numpy.std(val_scores))

# 5. Make predictions

test_set = pd.read_csv("X_test.csv")
x_test = test_set.drop('id', axis=1)
x_test = feature_selector.transform(x_test)
y_test = clf.predict(x_test)
Id = test_set['id']
df = pd.DataFrame(Id)
df.insert(1, "y", y_test)
df.to_csv('solution.csv', index=False)
