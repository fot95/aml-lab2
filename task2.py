import pandas as pd
import sys
import numpy
import copy

from sklearn.utils import column_or_1d
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score
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

cv_scores = cross_val_score(clf, x_train, y_train, scoring='balanced_accuracy', cv=10, n_jobs=10)
display_scores(cv_scores)

# 5. Make predictions

test_set = pd.read_csv("X_test.csv")
x_test = test_set.drop('id', axis=1)
y_test = clf.predict(x_test)
Id = test_set['id']
df = pd.DataFrame(Id)
df.insert(1, "y", y_test)
df.to_csv('solution.csv', index=False)
