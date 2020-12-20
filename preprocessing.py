import pandas as pd
import numpy as np
import seaborn as sns
import datetime
from sklearn.preprocessing import LabelBinarizer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import TruncatedSVD 


"""
    Functions to change the string based features to respective float values.
"""
def func(s):
    if(s=='CERTIFIED' or s=='Y'):
        return 1
    else:
        return 0

def func2(s):
    try:
        pp=datetime.datetime.strptime(s, "%m/%d/%Y")
    except:
        pp=datetime.datetime.strptime(s, "%d/%m/%Y")
    return pp

def func3(s):
    s=str(s)
    s=s.replace(',','')
    try:
        pp=float(s)
    except:
        pp=float(s[1:])
    return pp

def func4(s):
    if(s=='Year'):
        return 1
    elif(s=='Hour'):
        return 1920
    elif(s=='Week'):
        return 48
    elif(s=='Month'):
        return 12
    elif(s=='Bi-Weekly'):
        return 24

def func5(s):
    if(s>10**7):
        return s/2920
    else:
        return s

"""
    For sampling, we used under and over sampling.
"""
def undersample(df):
    cols = df.columns
    X = df[cols[1:]]
    Y = df[cols[0]]
    rus = RandomUnderSampler(sampling_strategy = {0: 5133, 1: 15000}, random_state=42)
    X,Y=rus.fit_resample(X,Y)
    df = pd.DataFrame()
    df[cols[0]] = Y
    df[cols[1:]] = X
    return df

"""
    Oversampling Technique.
"""
def oversample(df):
    df=df.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()
    cols = df.columns
    X = df[cols[:-1]]
    Y = df[cols[-1]]
    rus = RandomOverSampler(sampling_strategy = {0: 15000, 1: 15000}, random_state=42)
    X,Y=rus.fit_resample(X,Y)
    df = pd.DataFrame()
    df[cols[:-1]] = X
    df[cols[-1]] = Y
    return df

"""
    SVD Dimension Reduction
"""
def svd(df,k=30):
    cols = df.columns
    X = df[cols[1:]]
    Y = df[cols[0]]
    s = TruncatedSVD(n_components=k, random_state=0)
    X = s.fit_transform(X) # Fit and perform transformation
    return X, Y
"""
    Reading Original Dataset and removing redundant columns (almost no entries).
"""
df=pd.read_csv('original_dataset.csv')
df=df.dropna(axis=1, thresh=60000)
df.to_csv('dataset65cols.csv')

df=pd.read_csv('dataset65cols.csv')

#segregating columns manually having importance for this project
new_cols=['CASE_STATUS','CASE_SUBMITTED','DECISION_DATE','VISA_CLASS','SOC_CODE','FULL_TIME_POSITION','PERIOD_OF_EMPLOYMENT_START_DATE','PERIOD_OF_EMPLOYMENT_END_DATE','TOTAL_WORKER_POSITIONS','NEW_EMPLOYMENT','CONTINUED_EMPLOYMENT','CHANGE_PREVIOUS_EMPLOYMENT','NEW_CONCURRENT_EMPLOYMENT','CHANGE_EMPLOYER','AMENDED_PETITION','EMPLOYER_NAME','EMPLOYER_CITY','NAICS_CODE','AGENT_REPRESENTING_EMPLOYER','WAGE_RATE_OF_PAY_FROM_1','WAGE_UNIT_OF_PAY_1','H-1B_DEPENDENT','WILLFUL_VIOLATOR']

#segregating only H1-B visas entry
df=df[new_cols]
df=df[(df['CASE_STATUS']=='CERTIFIED') | (df['CASE_STATUS']=='DENIED')]
df=df[(df['VISA_CLASS']=='H-1B')]
df=df.dropna()

#preprocessing data so that it can be passed to different ML models

df['CASE_STATUS']=df['CASE_STATUS'].apply(func)
df['FULL_TIME_POSITION']=df['FULL_TIME_POSITION'].apply(func)
df['H-1B_DEPENDENT']=df['H-1B_DEPENDENT'].apply(func)
df['WILLFUL_VIOLATOR']=df['WILLFUL_VIOLATOR'].apply(func)
df['AGENT_REPRESENTING_EMPLOYER']=df['AGENT_REPRESENTING_EMPLOYER'].apply(func)
df['PERIOD_OF_EMPLOYMENT_START_DATE']=df['PERIOD_OF_EMPLOYMENT_START_DATE'].apply(func2)
df['PERIOD_OF_EMPLOYMENT_END_DATE']=df['PERIOD_OF_EMPLOYMENT_END_DATE'].apply(func2)
df['PERIOD_OF_EMPLOYMENT']=(df['PERIOD_OF_EMPLOYMENT_END_DATE']-df['PERIOD_OF_EMPLOYMENT_START_DATE']).astype('timedelta64[D]')
df['WAGE_RATE_OF_PAY_FROM_1']=df['WAGE_RATE_OF_PAY_FROM_1'].apply(func3)
df['WAGE_UNIT_OF_PAY_1']=df['WAGE_UNIT_OF_PAY_1'].apply(func4)
df['WAGE_RATE_OF_PAY_FROM_1']=df['WAGE_RATE_OF_PAY_FROM_1']*df['WAGE_UNIT_OF_PAY_1']
df['WAGE_RATE_OF_PAY_FROM_1']=df['WAGE_RATE_OF_PAY_FROM_1'].apply(func5)
df.drop('WAGE_UNIT_OF_PAY_1', inplace=True, axis=1)
df.drop('NAICS_CODE', inplace=True, axis=1)
df.drop('VISA_CLASS', inplace=True, axis=1)
df.drop('PERIOD_OF_EMPLOYMENT_END_DATE', inplace=True, axis=1)
df.drop('PERIOD_OF_EMPLOYMENT_START_DATE', inplace=True, axis=1)
df.drop('CASE_SUBMITTED', inplace=True, axis=1)
df.drop('DECISION_DATE', inplace=True, axis=1)
df.drop('TOTAL_WORKER_POSITIONS', inplace=True, axis=1)
df.to_csv('dataset16cols.csv', index=False)

#applying sampling techniques
df=pd.read_csv('dataset16cols.csv')
df=undersample(df)

#applying one-hot encoding
df=pd.get_dummies(df, prefix=['SOC_CODE', 'EMPLOYER_NAME', 'EMPLOYER_CITY'], columns=['SOC_CODE', 'EMPLOYER_NAME', 'EMPLOYER_CITY'])
df=df.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()

#applying SVD dimension reduction
X,Y = svd(df)
df = pd.DataFrame(data=X, columns=["col_"+str(i) for i in range(len(X[1]))])
df["Y"] = Y
df=oversample(df)

#creating final dataset that can be used to train models.
df.to_csv('Final_Dataset.csv', index=False)