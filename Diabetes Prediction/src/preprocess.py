from pandas.core.common import standardize_mapping
from utils import *

# Data Collection and Analysis
diabetes_dataset = pd.read_csv('diabetes.csv')

print(diabetes_dataset.head())

# Numnber of rows and columns
print(diabetes_dataset.shape)

#Getting the statsistical measures of the dataset
print(diabetes_dataset.describe())

print(diabetes_dataset['Outcome'].value_counts())   # 0 ---> Non Diabetic,  1 ---> Diabetic
print(diabetes_dataset.groupby('Outcome').mean())

X = diabetes_dataset.drop(columns = 'Outcome')
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

# Data Standardization
scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)
print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']

print(X)
print(Y)