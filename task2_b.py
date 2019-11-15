import pandas as pd
import sys
import numpy
import copy
import tensorflow
import os
import tempfile

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from tensorflow.keras import losses

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

x_train = x_train_init
y_train = y_train_init.y

classes = [0, 1, 2]
classes_pop = [0, 0, 0]
for i in list(y_train):
    classes_pop[i] += 1

total = classes_pop[0] + classes_pop[1] + classes_pop[2]

weight_for_0 = (1 / classes_pop[0])*(total)/3.0
weight_for_1 = (1 / classes_pop[1])*(total)/3.0
weight_for_2 = (1 / classes_pop[2])*(total)/3.0

class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))
print('Weight for class 2: {:.2f}'.format(weight_for_2))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
rmsprops = RMSprop()
optimizer=sgd

for dropout, epochs, batch_size, hidden_units in [(d, e, bs, hu)
                                                    for d  in [0, 0.1, 0.25, 0.5, 0.75, 0.9]
                                                    for e  in [50, 100, 150, 200, 250, 300, 400, 500, 700]
                                                    for bs in [100, 300, 500, 700, 1000, 1500, 2000]
                                                    for hu in [50, 75, 100, 125, 150, 175, 200, 300, 400, 600]
                                                 ]:
    model = Sequential()
    # Dense(hidden_units) is a fully-connected layer with hidden_units hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 200-dimensional vectors.
    model.add(Dense(hidden_units, activation='relu', input_dim=200))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', # that is the best the others are awful
                optimizer=optimizer, # https://keras.io/optimizers/
                metrics=['sparse_categorical_accuracy']) #TOD #TODO custom metric https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/O custom metric https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/

    y_train2 = y_train # keras.utils.to_categorical(y_train, num_classes=3)

    model.fit(x_train, y_train2,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            verbose=0)

    y_pred = model.predict_classes(x_train)
    # print(y_pred)

    # score = model.evaluate(x_train, y_train2, batch_size=128)
    # print(score)

    # 4. Classifier training + tuning

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
    for train_index, test_index in kf.split(x_train, y_train):
        X_train_i, X_val_i = x_train.iloc[train_index], x_train.iloc[test_index]
        Y_train_i, Y_val_i = y_train.iloc[train_index], y_train.iloc[test_index]

        model_i = Sequential()
        model_i.add(Dense(hidden_units, activation='relu', input_dim=200))
        model_i.add(Dropout(dropout))
        model_i.add(Dense(hidden_units, activation='relu'))
        model_i.add(Dropout(dropout))
        model_i.add(Dense(3, activation='softmax'))

        model_i.compile(loss='sparse_categorical_crossentropy', # that is the best the others are awful
                        optimizer=optimizer, # https://keras.io/optimizers/
                        metrics=['sparse_categorical_accuracy']) #TODO custom metric https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/

        classes_pop = [0, 0, 0]
        for i in list(y_train):
            classes_pop[i] += 1

        total = classes_pop[0] + classes_pop[1] + classes_pop[2]

        weight_for_0 = (1 / classes_pop[0])*(total)/3.0
        weight_for_1 = (1 / classes_pop[1])*(total)/3.0
        weight_for_2 = (1 / classes_pop[2])*(total)/3.0

        class_weight_i = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}

        model_i.fit(X_train_i, Y_train_i,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_i,
            verbose=0)

        Y_pred_val = model_i.predict_classes(X_val_i)

        test_s = balanced_accuracy_score(Y_val_i, Y_pred_val)

        val_scores.append(test_s)

    print(val_scores)
    print("mean: ", mean(val_scores))
    print("std: ", numpy.std(val_scores))
    print("^^^^^epochs:", epochs, "batch_size:", batch_size, "hidden_units:", hidden_units, "dropout:", dropout)

'''
# 5. Make predictions
test_set = pd.read_csv("X_test.csv")
x_test = test_set.drop('id', axis=1)
x_test = scaler.transform(x_test)
x_test = f_selector.transform(x_test)
y_test =  model.predict_classes(x_test)
Id = test_set['id']
df = pd.DataFrame(Id)
df.insert(1, "y", y_test)
df.to_csv('solution_keras_unscaled.csv', index=False)


test_set = pd.read_csv("X_test.csv")
x_test = test_set.drop('id', axis=1)
x_test_new = scaler.fit_transform(x_test)
cols = list(x_test.columns.values)
x_test = pd.DataFrame(data=x_test_new, columns=cols)
x_test = f_selector.transform(x_test)
y_test = model.predict_classes(x_test)
Id = test_set['id']
df = pd.DataFrame(Id)
df.insert(1, "y", y_test)
df.to_csv('solution_keras.csv', index=False)
'''
