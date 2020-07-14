from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
# x & y arrays filled from 0-9 & 0-4 respectively
'''
X:
01
23
45
67
89

y:
01234
'''
X, y = np.arange(10).reshape((5, 2)), range(5)
print('***Testing with random_state=None and shuffle=False***')
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)
# print how the dataset has split, test set is 33% of data,
# order of values is preserved by random_state default being False
print('--X_Train--')
print(x_train)
print('--X_Test--')
print(x_test)
print('--Y_Train--')
print(y_train)
print('--Y_Test--')
print(y_test)

print('***Testing with random_state=10 and shuffle=False***')
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10, shuffle=False)
# print how the dataset has split, test set is 33% of data,
# order of values should now be randomised, even if you run it multiple times this initial
# random order should be preserved because we have a fixed randomization of them
print('--X_Train--')
print(x_train)
print('--X_Test--')
print(x_test)
print('--Y_Train--')
print(y_train)
print('--Y_Test--')
print(y_test)

print('***KFold Split***')
dataset = range(16)  # array from 0 to 15
# splits being set to four means that it will split our data into four groups
# we have disabled shuffling and randomization to follow how these splits occur
KFCrossValidator = KFold(n_splits=4, shuffle=False)
KFdataset = KFCrossValidator.split(dataset)
print('{} {:^61} {}'.format('Round', 'Training set', 'Testing set'))
for iteration, data in enumerate(KFdataset, start=1):
    print('{:^9} {} {:^25}'.format(iteration, data[0], str(data[1])))
