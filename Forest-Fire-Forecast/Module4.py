import numpy
import pandas
from matplotlib import pyplot as plt
# region Making Printing more visible
from pandas.plotting._matplotlib import scatter_matrix

pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)
# endregion
# region Loading Our Data
filename = 'dataset/forestfires.csv'
names = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC',
         'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']
df = pandas.read_csv(filename, names=names)
# endregion
# region Analyzing Our Data
# print(pandas.isnull(df))
print("**************Data Shape**************")
print(df.shape)
print("**************Data Types**************")
print(df.dtypes)
print("**************Inspecting the head of the data**************")
print(df.head(1))
df.month.replace(('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'),
                 (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                 inplace=True)
df.day.replace(('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'), (1, 2, 3, 4, 5, 6, 7), inplace=True)
print("**************Inspecting the head of the data after replacement**************")
print(df.head(1))
print("**************Data Types Again**************")
print(df.dtypes)
print("**************Data Stats**************")
print(df.describe())
print("**************Correlation**************")
print(df.corr(method='pearson'))
# endregion

# region Visualzing Our data
# region Univariant
# histograms
df.hist(sharex=False, sharey=False, xlabelsize=3, ylabelsize=3)
plt.suptitle("Histograms", y=1.00, fontweight='bold')
plt.show()
# density
df.plot(kind='density', subplots=True, layout=(4, 4), sharex=False,
        fontsize=8)
plt.suptitle("Density", y=1.00, fontweight='bold')
plt.show()
# box and whisker plots
df.plot(kind='box', subplots=False, layout=(4, 4), sharex=False, sharey=False,
        fontsize=12)
plt.suptitle("Box and Whisker", y=1.00, fontweight='bold')
plt.show()
# endregion
# region Bivariant
# scatter plot matrix
scatter_matrix(df)
plt.suptitle("Scatter Matrix", y=1.00, fontweight='bold')
plt.show()
# correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(df.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = numpy.arange(0, 13, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.suptitle("Correlation Matrix", y=1.00, fontweight='bold')
plt.show()
# endregion
# endregion
