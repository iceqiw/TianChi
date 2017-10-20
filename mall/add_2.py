import pandas as pd

userMallSet2=pd.read_csv('evaluation_public.csv')
print(userMallSet2.count())
userMallSet=pd.read_csv('t_result.csv')
print(userMallSet.drop_duplicates().groupby(['rowid']).count())

