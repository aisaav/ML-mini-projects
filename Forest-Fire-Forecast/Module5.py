from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
print('***KFold Split***')
dataset=range(16) # array from 0 to 15
KFCrossValidator = KFold(n_splits=4, shuffle=False)
KFdataset = KFCrossValidator.split(dataset)
print('{} {:^61} {}'.format('Round', 'Training set', 'Testing set'))
for iteration, data in enumerate(KFdataset, start=1):
    print('{:^9} {} {:^25}'.format(iteration, data[0], str(data[1])))