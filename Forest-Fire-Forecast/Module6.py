import warnings

import pandas

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# region Making Printing more visible
warnings.filterwarnings('ignore')
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)
# endregion
# region Loading Our Data
filename = 'dataset/forestfires.csv'
names = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC',
         'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']
df = pandas.read_csv(filename, names=names)
# endregion
# region Preparing Data
df.month.replace(('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'),
                 (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), inplace=True)
df.day.replace(('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'), (1, 2, 3, 4, 5, 6, 7), inplace=True)
array = df.values
X = array[:, 0:12]
Y = array[:, 12]
# endregion

# region Folds and Scoring
num_folds = 10
seed = 7
scoring = 'max_error'
scoring2 = 'neg_mean_absolute_error'
scoring3 = 'r2'
scoring4 = 'neg_mean_squared_error'
# endregion

# region Spot-Check preliminary algorithms
models = []
models.append(('LR', LinearRegression()))  #
models.append(('LASSO', Lasso()))  #
models.append(('EN', ElasticNet()))  #
models.append(('Ridge', Ridge()))  #

models.append(('KNN', KNeighborsRegressor()))  #
models.append(('CART', DecisionTreeRegressor()))  #
models.append(('SVR', SVR()))  #
# endregion

# Evaluate models and print results
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    cv_results2 = cross_val_score(model, X, Y, cv=kfold, scoring=scoring2)
    cv_results3 = cross_val_score(model, X, Y, cv=kfold, scoring=scoring3)
    cv_results4 = cross_val_score(model, X, Y, cv=kfold, scoring=scoring4)
    msg = "%s: max error: %f , mean absolute error: %f, r2: %f, mean squared error: %f" % (name, cv_results.mean(),
                                                                                           -cv_results2.mean(),
                                                                                           cv_results3.mean(),
                                                                                           -cv_results4.mean())
    print(msg)
