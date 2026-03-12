import numpy as np
import pandas as pd
data = pd.read_csv("skin_cancer.csv")

# to see few rows and column
# to confirm the data is loaded correctly
print(data.head())
#
# # to get the number of row and column
print(data.shape)

# to get basic information about the data and datatype.
print(data.info())
print(data.describe())

# to get all the columns in form of a list
print(data.columns)

#to see missing values
print(data.isnull().sum())
#
data['days_of_treatment'] = (
        pd.to_datetime(data['EAS Decision date'], format='mixed') -
        pd.to_datetime(data['Diagnosis date'], format='mixed')
).dt.days
print(data['days_of_treatment'])

data = data.drop(
    ['Patient identifier', 'Date of Birth', 'Cancer Group','Diagnosis histology',
     'EAS Decision date', 'Diagnosis date'], axis=1, errors='ignore')

# # To make a Target column from DCD date
data['DCD date'] = data['DCD date'].notna().astype(int)
print(data['DCD date'].value_counts())
# #
# # # to group together
conditions = [
    data['Diagnosis ICD code'].str.startswith(('C43.0','C43.1','C43.2','C43.3','C43.4')),
    data['Diagnosis ICD code'].str.startswith('C43.5'),
    data['Diagnosis ICD code'].str.startswith(('C43.6','C43.7','C43.8')),
    data['Diagnosis ICD code'].str.startswith('C43.9')
]

choices = ["Head_Neck","Trunk","Limb","Unspecified"]
data['ICD_group'] = np.select(conditions,choices,default='Unspecified')
print(data['ICD_group'].value_counts())
data = data.drop(['Diagnosis ICD code'], axis=1, errors='ignore')
print(data.columns.tolist())

# # to change the column names according to my preference
data = (data[
['Age at diagnosis', 'Gender', 'DCD date', 'EAS regimen', 'Intent', 'Line of treatment', 'T', 'N', 'M', 'EAS Performance status', 'Letter Diagnosis',
'Letter Summary', 'Letter Plan', 'days_of_treatment', 'ICD_group']]).rename(columns={"Age at diagnosis" : "Age", "DCD date" : "Alive or dead",
                                                                                     "days_of_treatment" : "Days of treatment","ICD_group" : "ICD group"})
print(data.head())

# To change the clean data into csv file for ML
data.to_csv('melanoma_clean.csv', index=False)

df = pd.read_csv("melanoma_clean.csv")
print(df['ICD group'].unique())
print(df['Alive or dead'].head())


