import pandas
# Program for data loading from course module-3
filename = 'forestfires.csv'
# Column data names
col_names = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC',
         'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']
# Create dataframe
df = pandas.read_csv(filename, names=col_names)
# Checks for null/missing values in the dataset
print(pandas.isnull(df))
