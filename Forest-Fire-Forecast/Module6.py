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
# region Loading Our Data - copied from Module 3
filename = 'forestfires.csv'
names = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC',
         'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']
df = pandas.read_csv(filename, names=names)
# endregion
# region Preparing Data- copied from Module 4
df.month.replace(('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'),
                 (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), inplace=True)
df.day.replace(('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'), (1, 2, 3, 4, 5, 6, 7), inplace=True)
array = df.values
X = array[:, 0:12]  # features
Y = array[:, 12]  # labels
# endregion

# region Folds and Scoring, variables needed for algorithm training
num_folds = 10  # k-value
seed = 7  # random seed
scoring = 'max_error'
scoring2 = 'neg_mean_absolute_error'
scoring3 = 'r2'
scoring4 = 'neg_mean_squared_error'
# endregion

# region Spot-Check preliminary algorithms
# refer to scikit-learn documentation if something is confusing
# Array containing all instances of models I would like to train on
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('Ridge', Ridge()))

models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
# endregion

# Evaluate models and print results
results = []
names = []
for name, model in models:
    '''
    look through all models
    in each iteration define a kfold instance w/ set K value & random state
    Cross_val_score function trains and evaluates the model in a single call
    Four calls for the four scoring methods we would like to see
    cv_results will be an array with 10 different values for each kfold drawn, we are interested in the mean of each 
    cv_result array- we take the mean bc it is how the kfold validation is defined
    take negatives of mean absolute and mean squared errors because scikit learn has defined these as negative
    longer discussion on this @ http://bit.ly/2Dld0KM 
    This will ultimately tell us what algoritm is best for further investigation
    '''
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
