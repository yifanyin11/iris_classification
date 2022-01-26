import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict


# ## Data Preprocessing 

df = pd.read_csv(r'iris.data',header = None, names = ['Sepal_length','Sepal_width','Petal_length','Petal_width','Name'])
# label encoding
# converting type of columns to 'category'
df['Name'] = df['Name'].astype('category')
# Assigning numerical values and storing in another column
df['Name_Cat'] = df['Name'].cat.codes
# drop original labels
df = df.drop(columns='Name')
# separate input and output data
data = df.values
X = data[:,0:np.size(data,axis=1)-1]
Y = data[:,-1]
# standardize
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
feature_num = np.size(X, axis=1)


# ## Logistic Regression

# tune hyperparameters with 5-fold CV
param_grid_lr = {'solver':('liblinear','newton-cg','lbfgs','saga','sag'), 
              'C':[0.01, 0.1, 1, 10, 50, 100],
              'max_iter':[500]}
grid_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, refit=True,verbose=0)
grid_lr.fit(X_train, Y_train)
print(grid_lr.best_estimator_)

clf_lr = LogisticRegression(random_state=0, C=50, max_iter=500, solver='newton-cg').fit(X_train, Y_train)
clf_lr.fit(X_train, Y_train)
test_err1 = clf_lr.score(X_test, Y_test)
print(test_err1)


# ## SVM

# tune hyperparameters
param_grid_svm = {'kernel':('linear','poly', 'rbf','sigmoid'), 
              'gamma': [100, 10, 1, 0.1, 0.01, 0.001],
              'C':[0.01, 0.1, 1, 10, 50, 100, 200]}

grid_svm = GridSearchCV(SVC(), param_grid_svm, refit=True,verbose=0)
grid_svm.fit(X_train, Y_train)
print(grid_svm.best_estimator_)

clf_svc = SVC(C=50, gamma=0.1, kernel='linear')
clf_svc.fit(X_train, Y_train)
test_err2 = clf_svc.score(X_test, Y_test)
print(test_err2)


# ## Random Forests

# tune hyperparameters
param_grid_rf = {'n_estimators': [5, 10, 100, 500], 
                'max_depth': [2, 5, 10],
              'max_features': ('sqrt','log2')}

grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, n_jobs= -1, error_score= 0,verbose=0)
grid_rf.fit(X_train, Y_train)
print(grid_rf.best_estimator_)

clf_rf = RandomForestClassifier(max_depth=5, max_features='sqrt', n_estimators=10)
clf_rf.fit(X_train, Y_train)
test_err3 = clf_rf.score(X_test, Y_test)
print(test_err3)

RANDOM_STATE=10
ensemble_clfs = [
    (
        "RF, max_features='sqrt'",
        RandomForestClassifier(
            warm_start=True,
            oob_score=True,
            max_features='sqrt',
            max_depth=10,
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "RF, max_features='log2'",
        RandomForestClassifier(
            warm_start=True,
            max_features='log2',
            oob_score=True,
            max_depth=10,
            random_state=RANDOM_STATE,
        ),
    ),
]

error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of 'n_estimators' values to explore.
min_estimators = 30
max_estimators = 200

min_error = np.inf

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators+1):
        clf.set_params(n_estimators=i, max_depth=j)
        clf.fit(X_train, Y_train)
        oob_error = 1 - clf.oob_score_
        if oob_error < min_error:
            min_error = oob_error
            best_model = (label, i, min_error)

print(best_model)

print(best_model)
clf_rf = RandomForestClassifier(max_depth=10, max_features='sqrt', n_estimators=48)
clf_rf.fit(X_train, Y_train)
test_err3 = clf_rf.score(X_test, Y_test)
print(test_err3)

